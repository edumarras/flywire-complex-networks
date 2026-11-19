from pathlib import Path
import gzip
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
FEATURES = ROOT / 'reports' / 'f2' / 'features'
OUT = ROOT / 'reports' / 'f2' / 'audit_target_leakage'
OUT.mkdir(parents=True, exist_ok=True)

CLASSIFICATION = RAW / 'classification.csv.gz'
RAW_CONNECTIONS = RAW / 'connections_princeton.csv.gz'
PROCESSED_EDGES = PROCESSED / 'flywire_fafb_v783_edges.csv.gz'
TOPO_TABLE = FEATURES / 'node_topological_features.csv.gz'
TOPO_MANIFEST = FEATURES / 'topological_feature_manifest.csv'
BIO_TABLE = FEATURES / 'node_biological_features.csv.gz'
TARGET_TABLE = FEATURES / 'node_targets.csv.gz'

TOPOLOGY_LINEAGE_SCRIPTS = [
    ROOT / 'scripts' / 'f1_02_build_graph.py',
    ROOT / 'scripts' / 'f2_01_structural_metrics.py',
    ROOT / 'scripts' / 'f2_03_communities.py',
    ROOT / 'scripts' / 'f2_04_modular_roles.py',
    ROOT / 'scripts' / 'f2_06_build_topological_features.py',
]

FORBIDDEN_TOPOLOGY_FIELDS = {
    'super_class','flow','class','sub_class','hemilineage','nerve','group',
    'primary_type','additional_type(s)','cell_type','resolved_type','neuropil',
    'side','nt_type','nt_type_score','pos_x','pos_y','pos_z','soma_x','soma_y','soma_z'
}
FORBIDDEN_SOURCE_FILES = {
    'classification.csv.gz','consolidated_cell_types.csv.gz','node_targets.csv.gz',
    'node_biological_features.csv.gz'
}
EXPECTED_TARGET_COLUMNS = [
    'root_id','flow','super_class','class','sub_class','hemilineage','side','nerve'
]
EXPECTED_PROCESSED_EDGE_COLUMNS = ['pre_root_id','post_root_id','weight']


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_gzip_payload(path):
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(path):
    if not path.exists():
        raise FileNotFoundError(f'Arquivo necessário ausente: {path}')


def main():
    print('F2.13 — Proveniência do target e auditoria de leakage')
    for path in [CLASSIFICATION, RAW_CONNECTIONS, PROCESSED_EDGES, TOPO_TABLE,
                 TOPO_MANIFEST, BIO_TABLE, TARGET_TABLE]:
        require(path)

    # 1. Freeze local provenance by hash
    hash_rows = []
    for label, path in [
        ('classification', CLASSIFICATION),
        ('raw_connections', RAW_CONNECTIONS),
        ('processed_edges', PROCESSED_EDGES),
        ('topological_features', TOPO_TABLE),
        ('biological_features', BIO_TABLE),
        ('target_table', TARGET_TABLE),
    ]:
        row = {
            'artifact': label,
            'relative_path': path.relative_to(ROOT).as_posix(),
            'size_bytes': path.stat().st_size,
            'sha256_file': sha256_file(path),
            'sha256_uncompressed_payload': sha256_gzip_payload(path) if path.suffix == '.gz' else '',
        }
        hash_rows.append(row)
    hashes = pd.DataFrame(hash_rows)
    hashes.to_csv(OUT / 'source_hashes.csv', index=False)

    # 2. Target schema/counts/flow relationship
    cls = pd.read_csv(CLASSIFICATION, low_memory=False)
    if list(cls.columns) != EXPECTED_TARGET_COLUMNS:
        raise RuntimeError(f'Schema classification inesperado: {list(cls.columns)}')
    if cls.root_id.duplicated().any() or cls.super_class.isna().any():
        raise RuntimeError('classification possui root_id duplicado ou target ausente')

    target_counts = cls.super_class.value_counts().rename_axis('super_class').reset_index(name='count')
    target_counts['fraction'] = target_counts['count'] / len(cls)
    target_counts.to_csv(OUT / 'target_counts.csv', index=False)

    flow_table = pd.crosstab(cls.super_class, cls.flow, dropna=False)
    flow_table.to_csv(OUT / 'superclass_by_flow.csv')
    dominant_flow_fraction = flow_table.max(axis=1) / flow_table.sum(axis=1)
    n_perfect_flow = int((dominant_flow_fraction == 1.0).sum())

    # 3. Graph matter: raw has metadata; processed graph must not.
    raw_header = pd.read_csv(RAW_CONNECTIONS, nrows=0).columns.tolist()
    processed_header = pd.read_csv(PROCESSED_EDGES, nrows=0).columns.tolist()
    if processed_header != EXPECTED_PROCESSED_EDGE_COLUMNS:
        raise RuntimeError(f'Processed edge schema inesperado: {processed_header}')
    leaked_raw_metadata = sorted(set(processed_header) & {'neuropil','nt_type'})
    if leaked_raw_metadata:
        raise RuntimeError(f'Metadados de aresta chegaram ao grafo processado: {leaked_raw_metadata}')

    # 4. Frozen topology whitelist
    topo_header = pd.read_csv(TOPO_TABLE, nrows=0).columns.tolist()
    topo_manifest = pd.read_csv(TOPO_MANIFEST)
    topo_features = [c for c in topo_header if c != 'root_id']
    manifest_features = topo_manifest.feature.tolist()
    if set(topo_features) != set(manifest_features):
        raise RuntimeError('Manifesto e tabela topológica divergem')
    forbidden_in_topo = sorted(set(topo_features) & FORBIDDEN_TOPOLOGY_FIELDS)
    if forbidden_in_topo:
        raise RuntimeError(f'Campos proibidos na tabela REDE: {forbidden_in_topo}')

    # 5. Target separation
    bio_header = pd.read_csv(BIO_TABLE, nrows=0).columns.tolist()
    target_header = pd.read_csv(TARGET_TABLE, nrows=0).columns.tolist()
    if 'super_class' in topo_header or 'super_class' in bio_header:
        raise RuntimeError('super_class entrou em uma tabela de features')
    if target_header != ['root_id','super_class']:
        raise RuntimeError(f'node_targets schema inesperado: {target_header}')

    # 6. Static upstream dependency audit
    lineage_rows = []
    missing_scripts = []
    hard_fail = False
    for path in TOPOLOGY_LINEAGE_SCRIPTS:
        if not path.exists():
            missing_scripts.append(path.name)
            lineage_rows.append({'script': path.name, 'exists': False,
                                 'forbidden_source_references': '', 'passes': False})
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        hits = sorted(name for name in FORBIDDEN_SOURCE_FILES if name in text)
        passes = len(hits) == 0
        hard_fail = hard_fail or not passes
        lineage_rows.append({'script': path.name, 'exists': True,
                             'forbidden_source_references': ';'.join(hits), 'passes': passes})
    pd.DataFrame(lineage_rows).to_csv(OUT / 'topological_lineage_audit.csv', index=False)

    # 7. Explicit forbidden-field audit
    forbidden_rows = []
    for field in sorted(FORBIDDEN_TOPOLOGY_FIELDS):
        present_table = field in topo_features
        present_manifest = field in manifest_features
        forbidden_rows.append({
            'field': field,
            'present_in_topological_table': present_table,
            'present_in_topological_manifest': present_manifest,
            'status': 'FAIL' if present_table or present_manifest else 'PASS'
        })
    forbidden_df = pd.DataFrame(forbidden_rows)
    forbidden_df.to_csv(OUT / 'forbidden_columns_audit.csv', index=False)
    if (forbidden_df.status == 'FAIL').any():
        raise RuntimeError('Forbidden-field audit falhou')

    technical_pass = (
        not forbidden_in_topo and not hard_fail and not leaked_raw_metadata
        and 'super_class' not in bio_header and 'super_class' not in topo_header
    )

    summary = {
        'audit_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'target': {
            'rows': int(len(cls)),
            'superclasses': int(cls.super_class.nunique()),
            'flows': int(cls.flow.nunique()),
            'classification_sha256': hashes.loc[hashes.artifact == 'classification','sha256_file'].iloc[0],
            'classification_uncompressed_sha256': hashes.loc[hashes.artifact == 'classification','sha256_uncompressed_payload'].iloc[0],
            'superclass_counts': dict(zip(target_counts.super_class, target_counts['count'].astype(int))),
        },
        'flow_relationship': {
            'superclasses_with_single_flow_only': n_perfect_flow,
            'total_superclasses': int(len(flow_table)),
            'note': 'flow é upstream de super_class; a análise condicionada por flow é um controle complementar.'
        },
        'graph_lineage': {
            'raw_connection_columns': raw_header,
            'processed_edge_columns': processed_header,
            'neuropil_used_as_topological_feature': False,
            'edge_nt_type_used_as_topological_feature': False,
        },
        'frozen_topology': {
            'n_features': len(topo_features),
            'features': topo_features,
            'forbidden_fields_present': forbidden_in_topo,
        },
        'code_lineage': {
            'complete': len(missing_scripts) == 0,
            'missing_scripts': missing_scripts,
            'checks': lineage_rows,
        },
        'technical_leakage_conclusion': {
            'pass': technical_pass,
            'statement': (
                'Nenhum target/campo taxonômico direto está na tabela topológica congelada; '
                'o grafo processado contém apenas IDs e peso sináptico agregado. Isso sustenta '
                'ausência de leakage DIRETO, mas não prova independência conceitual entre '
                'super_class e anatomia conectômica.'
            )
        },
        'next_required_controls': [
            'fechar versão/proveniência das anotações hierárquicas',
            'documentar definição de cada super_class',
            'predizer super_class condicionada a flow',
            'rede real vs degree-preserving rewired',
            'baseline anatômico por neuropil',
        ]
    }
    (OUT / 'target_leakage_audit_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    lines = [
        'FlyWire FAFB v783 — auditoria de proveniência do target e leakage', '',
        f'rows: {len(cls)}',
        f'superclasses: {cls.super_class.nunique()}',
        f'flows: {cls.flow.nunique()}',
        f'classification sha256: {summary["target"]["classification_sha256"]}',
        f'classification payload sha256: {summary["target"]["classification_uncompressed_sha256"]}', '',
        'GRAFO REDE:',
        f'raw columns: {raw_header}',
        f'processed columns: {processed_header}',
        'neuropil usado como feature topológica: NÃO',
        'nt_type da aresta usado como feature topológica: NÃO', '',
        f'features topológicas congeladas: {len(topo_features)}',
        'campos proibidos presentes: nenhum', '',
        f'FLOW: {n_perfect_flow}/{len(flow_table)} superclasses aparecem em um único flow.',
        'A relação flow-super_class motiva o controle condicionado por flow.', '',
        'CONCLUSÃO TÉCNICA:',
        ('PASS — não há evidência de leakage DIRETO.' if technical_pass
         else 'AUDIT INCOMPLETO/FAIL — revisar saída.'),
        'Isto NÃO estabelece independência conceitual entre super_class e anatomia/conectividade.'
    ]
    if missing_scripts:
        lines += ['', 'Scripts upstream ausentes nesta cópia: ' + ', '.join(missing_scripts)]
    (OUT / 'target_leakage_audit_summary.txt').write_text('\n'.join(lines), encoding='utf-8')

    print('\nTarget counts:')
    print(target_counts.to_string(index=False))
    print('\nsuper_class × flow:')
    print(flow_table.to_string())
    print('\nDIRECT TARGET LEAKAGE:', 'PASS — não detectado' if technical_pass else 'INCOMPLETO/FAIL')
    print('CONCEPTUAL INDEPENDENCE: ainda NÃO estabelecida')
    print('Output:', OUT)


if __name__ == '__main__':
    main()
