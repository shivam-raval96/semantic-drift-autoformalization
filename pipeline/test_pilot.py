"""Run: python -m pytest test_pilot.py -q   (or just: python test_pilot.py)
Every test states a fact you can verify by hand."""
from laws import LAWS, parse_law_str, render_law
from magma import satisfies, implication_status, classify_relation, find_countermodel
from dataset import generate
from pilot import MockBackend, run, score

def test_nand_separates_comm_from_assoc():
    nand = [[1,1],[1,0]]
    assert satisfies(nand, LAWS['comm'], 2)
    assert not satisfies(nand, LAWS['assoc'], 2)

def test_left_projection_separates_assoc_from_comm():
    lp = [[0,0],[1,1]]
    assert satisfies(lp, LAWS['assoc'], 2) and satisfies(lp, LAWS['lproj'], 2)
    assert not satisfies(lp, LAWS['comm'], 2)

def test_sps_proposal_example_table():
    # the exact three rows from the SPS proposal's "Example Semantic Difference"
    assert classify_relation(LAWS['comm'], LAWS['refl'])[0].startswith('weaker')
    assert classify_relation(LAWS['refl'], LAWS['triv'])[0].startswith('stronger')
    assert classify_relation(LAWS['comm'], LAWS['assoc'])[0] == 'incomparable'

def test_certificates_are_real():
    label, cert = implication_status(LAWS['comm'], LAWS['assoc'])
    assert label == 'non-implication'
    n, table = cert['size'], cert['table']
    assert satisfies(table, LAWS['comm'], n) and not satisfies(table, LAWS['assoc'], n)

def test_known_implications():
    assert implication_status(LAWS['triv'], LAWS['medial'])[0] == 'implies (known)'
    assert implication_status(LAWS['assoc'], LAWS['refl'])[0] == 'implies (known)'

def test_dataset_gold_is_certified():
    from laws import apply_substitution, parse_term
    for item in generate(0):
        if item['family'] != 'I':
            continue
        p, c = LAWS[item['premise_lid']], LAWS[item['conclusion_lid']]
        cert = item['certificate']
        if item['gold'] == 'does_not_imply':
            t, n = cert['table'], cert['size']
            assert satisfies(t, p, n) and not satisfies(t, c, n)
        elif cert['route'] == 'substitution instance':
            # re-verify: applying the stored substitution to the premise must yield
            # exactly the conclusion (up to orienting '=')
            sigma = {v: parse_term(s) for v, s in cert['substitution'].items()}
            al, ar = apply_substitution(p.lhs, sigma), apply_substitution(p.rhs, sigma)
            assert (al, ar) in ((c.lhs, c.rhs), (c.rhs, c.lhs)), item['item_id']
        else:
            assert cert['route'] == 'construction-known'
            assert c.lid == 'refl' or p.lid == 'triv' or p.lid == c.lid

def test_substitution_instance_route():
    # flexibility is associativity with z := x; commutativity is an instance of
    # the constant-product law (z := y, w := x)
    label, cert = implication_status(LAWS['assoc'], LAWS['flex'])
    assert label == 'implies (known)' and cert['route'] == 'substitution instance'
    label, cert = implication_status(LAWS['const'], LAWS['comm'])
    assert label == 'implies (known)' and cert['route'] == 'substitution instance'
    # and NOT the other way around: flex does not imply assoc
    label, cert = implication_status(LAWS['flex'], LAWS['assoc'])
    assert label == 'non-implication'

def test_batch_certifier_agrees_with_per_pair():
    # spot-check certify_all against implication_status on a mixed sample
    from magma import certify_all
    sample = [LAWS[l] for l in ('comm', 'assoc', 'refl', 'triv', 'flex', 'lproj')]
    batch = certify_all(sample)
    for p in sample:
        for c in sample:
            assert batch[(p.lid, c.lid)][0] == implication_status(p, c)[0], (p.lid, c.lid)

def test_drift_targets_never_equivalent_to_intended():
    # Family A gold integrity: every (intended, drift-target) pair must be separated
    # by a countermodel in at least one direction, else 'drift' items could score
    # as faithful (equivalent*)
    from dataset import certified_pairs, CORE
    from laws import drift_moves
    certs = certified_pairs()
    for lid in CORE:
        for drifted, move in drift_moves(LAWS[lid]):
            fwd, bwd = certs[(lid, drifted.lid)][0], certs[(drifted.lid, lid)][0]
            assert 'non-implication' in (fwd, bwd), (lid, drifted.lid, move, fwd, bwd)

def test_variable_role_swap_distinct_where_promised():
    # assoc, labsorb and medial must carry a varswap target distinct from their
    # neighbor_confusion target (the previously-documented dedupe limitation)
    from laws import drift_moves
    for lid in ('assoc', 'labsorb', 'medial'):
        moves = {mv: law.lid for law, mv in drift_moves(LAWS[lid])}
        assert 'variable_role_swap' in moves and 'neighbor_confusion' in moves, lid
        assert moves['variable_role_swap'] != moves['neighbor_confusion'], lid

def test_family_sizes_and_control_ratio():
    from collections import Counter
    counts = Counter(d['family'] for d in generate(0))
    assert 50 <= counts['A'] <= 100, counts   # spec: 50-100 Family A pairs
    experimental = counts['A'] + counts['B']
    assert experimental / 5 <= counts['D'] <= experimental / 3, counts  # ~1 control per 4

def test_core_laws_have_full_template_sets():
    from dataset import CORE
    from laws import NL_TEMPLATES
    for lid in CORE:
        assert set(NL_TEMPLATES[lid]) == {'canonical', 'paraphrase', 'instance', 'distractor'}, lid

def test_pipeline_end_to_end_and_faithful_when_no_drift():
    class Perfect(MockBackend):
        SYNTAX_P = 0.0
        DRIFT_P = {k: 0.0 for k in MockBackend.DRIFT_P}
        B_DRIFT_P = 0.0
        IMPL_ERR = {k: 0.0 for k in MockBackend.IMPL_ERR}
        def judge_implication(self, item): return item['gold']
    scored = score(run(generate(0), Perfect(0)))
    bad = [s for s in scored if s['verdict'] not in ('faithful', 'correct')]
    assert not bad, bad[:3]

def test_api_backend_offline_with_fake_transport():
    # exercise the real ApiBackend code path (prompt build, fence stripping, judgment
    # normalization) without network: transport echoes canned model behavior
    from pilot import ApiBackend, build_prompt
    def fake_transport(prompt):
        if 'does_not_imply' in prompt:  # implication prompt lists both options
            return 'implies'
        if 'commutative' in prompt or 'either order' in prompt or 'b combined with a' in prompt:
            return '```\nx * y = y * x\n```'
        return 'x * (y ='  # everything else: an unparsable law
    b = ApiBackend.__new__(ApiBackend)
    b.model, b.name, b.provider = 'fake', 'api:fake', 'anthropic'
    b.deterministic, b.max_output_tokens, b.transport = True, 300, fake_transport
    data = [d for d in generate(0)
            if d['family'] == 'B' and d['intended_lid'] == 'comm' and d['register'] != 'distractor']
    impl = [d for d in generate(0) if d['family'] == 'I'][:4]
    scored = score(run(data + impl, b))
    for s in scored:
        if s['family'] == 'B':
            assert s['verdict'] == 'faithful', (s['item_id'], s['verdict'])
        else:
            expected = 'correct' if s['gold'] == 'implies' else 'incorrect'
            assert s['verdict'] == expected

def test_provider_routing():
    from pilot import detect_provider
    assert detect_provider('claude-opus-4-8') == ('anthropic', False)   # temp rejected -> seeds
    assert detect_provider('claude-haiku-4-5') == ('anthropic', True)
    assert detect_provider('gpt-4o-mini') == ('openai', True)
    assert detect_provider('openai/gpt-4o-mini') == ('openrouter', True)
    assert detect_provider('anthropic/claude-opus-4-8') == ('openrouter', False)
    assert detect_provider('meta-llama/llama-3.3-70b-instruct') == ('openrouter', True)

def test_missing_key_fails_loud():
    import os
    from pilot import ApiBackend
    saved = os.environ.pop('OPENROUTER_API_KEY', None)
    try:
        try:
            ApiBackend('meta-llama/llama-3.3-70b-instruct')
            assert False, 'expected RuntimeError'
        except RuntimeError as e:
            assert 'OPENROUTER_API_KEY' in str(e)
    finally:
        if saved is not None:
            os.environ['OPENROUTER_API_KEY'] = saved

def test_prompts_never_leak_gold():
    # guardrail: drift targets, gold labels, certificates must not reach any prompt
    from pilot import build_prompt
    for item in generate(0):
        prompt = build_prompt(item)
        if item['family'] == 'A':
            assert item['drift_law'] not in prompt
            assert item['drift_move'] not in prompt
        if item['family'] == 'I':
            assert 'table' not in prompt and 'certificate' not in prompt
            # both answer options must appear (no asymmetric hint toward gold)
            assert 'implies' in prompt and 'does_not_imply' in prompt

def test_z3_backend_mirrors_brute_force():
    # optional backend: skip silently if z3 isn't installed
    from z3_check import z3_available, find_countermodel_z3
    if not z3_available():
        return
    cm = find_countermodel_z3(LAWS['comm'], LAWS['assoc'], sizes=range(2, 4))
    assert cm is not None
    n, table = cm
    assert satisfies(table, LAWS['comm'], n) and not satisfies(table, LAWS['assoc'], n)
    # agrees with brute force on a known implication (no countermodel at any size <=3)
    assert find_countermodel_z3(LAWS['assoc'], LAWS['flex'], sizes=range(2, 4)) is None

def test_prompt_freeze():
    """Pre-registration, executable: every fixed prompt and NL template is pinned by
    hash. If this fails you CHANGED THE INSTRUMENT — that is sometimes right (e.g. a
    signed-off template repair), but must be a conscious act: update the hash in the
    same commit and note the change in the findings file. Results produced under
    different hashes are different experiments and must not be pooled."""
    import hashlib, json
    from pilot import PROMPTS
    from telephone import INFORMALIZE_PROMPT
    from selfreport import JUDGE_PROMPT
    from laws import NL_TEMPLATES
    frozen = {'prompts': PROMPTS, 'informalize': INFORMALIZE_PROMPT,
              'judge': JUDGE_PROMPT, 'templates': NL_TEMPLATES}
    h = hashlib.sha256(json.dumps(frozen, sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()[:16]
    assert h == 'e41a158babcfec38', f'instrument changed: {h}'

def test_renderer_freeze():
    """Audit repair 2026-07-07: the freeze perimeter now covers stimulus RENDERERS
    (etp/ladder NL generation), not only NL_TEMPLATES/prompts. Changing a renderer
    changes the experiment: update the hash consciously, regenerate data files,
    note in findings. (The gate-passed A2 short-name swap will change this hash
    when applied — that is the mechanism working.)"""
    import hashlib, inspect
    import etp_items, ladder_items
    parts = [inspect.getsource(etp_items.term_nl), inspect.getsource(etp_items.law_nl),
             repr(sorted(etp_items.VAR_NAMES.items())),
             inspect.getsource(ladder_items.rung_surfaces),
             inspect.getsource(ladder_items.term_entity),
             repr(sorted(ladder_items.ENTITY.items()))]
    h = hashlib.sha256('\n'.join(parts).encode()).hexdigest()[:16]
    assert h == 'b23fa0da1704491b', f'stimulus renderer changed: {h}'

def test_strict_parser_and_three_way_verdict():
    from laws import parse_term, ParseError
    parse_term('x * y * z')  # lenient: documented left-assoc convention
    try:
        parse_term('x * y * z', strict=True)
        assert False, 'strict should reject ambiguous chains'
    except ParseError:
        pass
    from pilot import verdict_from_relation
    assert verdict_from_relation('equivalent') == 'faithful'
    assert verdict_from_relation('equivalent*') == 'unresolved (<=4)'
    assert verdict_from_relation('incomparable') == 'drift: incomparable'

def test_prompt_freeze_v2():
    """Confirmatory bank (handcheck-2026-07-05 sign-offs) pinned separately from
    the frozen v1 exploratory strings. Changing it = conscious act, same protocol."""
    import hashlib, json
    from templates_v2 import bank
    b = {lid: {r: s for r, s in regs.items()} for lid, regs in bank().items()}
    h = hashlib.sha256(json.dumps(b, sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()[:16]
    assert h == '048d8ec6e4a35108', f'confirmatory instrument changed: {h}'

def test_v2_bank_implements_handcheck():
    from templates_v2 import bank
    b = bank()
    assert 'canonical' not in b['labsorb'] and 'canonical' not in b['rabsorb']  # dropped
    assert b['unipot']['canonical'][1] == 'obscure-name-stratum'
    assert all('distractor' in regs for regs in b.values())  # well-posed v2 register, all 10
    assert 'combine that result' in b['rabsorb']['paraphrase'][0]  # rewrite landed
    from dataset import generate_v2
    data = generate_v2(0)
    B2 = [d for d in data if d['family'] == 'B']
    assert all(d.get('template_version') == 'v2' for d in B2)
    # distractor items exist in B and carry gold = intended (well-posed by design)
    assert sum(1 for d in B2 if d['register'] == 'distractor') == 10

def test_probe_split_no_leakage():
    from dataset import probe_split
    rows = [d for d in generate(0) if d['task'] in ('translation', 'transcription')]
    tr, te = probe_split(rows, mode='register', held_out='paraphrase')
    assert te and all(r['register'] == 'paraphrase' for r in te)
    assert not ({r['surface'] for r in tr} & {r['surface'] for r in te})
    tr2, te2 = probe_split(rows, mode='law', seed=0)
    assert te2 and not ({r['intended_lid'] for r in tr2} & {r['intended_lid'] for r in te2})

def test_telephone_offline():
    from telephone import run_chain, classify_records, canonical
    # fake model: echoes laws faithfully except it corrupts comm -> assoc once
    def transport(prompt):
        if prompt.startswith('Describe'):
            law = prompt.split('Law: ')[1].split('\n')[0]
            return f"SENTENCE[{law}]"
        law = prompt.split('equation: ')[1].removeprefix('SENTENCE[').removesuffix(']')
        if law == '(x * y) = (y * x)':
            return 'x * (y * z) = (x * y) * z'  # one drift, then stable
        return law
    recs = run_chain(transport, 'comm', canonical('x * y = y * x'), hops=3)
    for r in recs:
        r['backend'] = 'fake'
    classify_records(recs)
    assert [r['alive'] for r in recs] == [False, False, False]
    assert recs[0]['rel_to_prev'] == 'incomparable'          # the drift hop
    assert recs[1]['rel_to_prev'].startswith('equivalent')   # stable afterwards
    recs2 = run_chain(transport, 'idem', canonical('x * x = x'), hops=3)
    for r in recs2:
        r['backend'] = 'fake'
    classify_records(recs2)
    assert all(r['alive'] for r in recs2)
    # death on unparsable output
    recs3 = run_chain(lambda p: 'x * (y =' if 'equation' in p else 'Describe: ok', 'comm',
                      canonical('x * y = y * x'), hops=3)
    assert len(recs3) == 1 and 'death' in recs3[0]

def test_etp_ids_consistent_with_recorded_strings():
    # offline re-check of [VERIFY-ETP]: laws.py's etp ids must agree with the ETP
    # equation strings recorded (and network-verified) in verify_etp.ETP_STRINGS;
    # laws without an id must be outside the ETP enumeration (> 4 op applications)
    from verify_etp import ETP_STRINGS, parse_etp, same_law
    for lid, law in LAWS.items():
        if lid in ETP_STRINGS:
            n, s = ETP_STRINGS[lid]
            assert law.etp_node == f'Eq{n}', (lid, law.etp_node, n)
            assert same_law((law.lhs, law.rhs), parse_etp(s)), (lid, s)
        else:
            assert law.etp_node is None and law.n_ops > 4, lid

if __name__ == '__main__':
    import sys
    g = dict(globals())
    fails = 0
    for name, fn in g.items():
        if name.startswith('test_'):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
