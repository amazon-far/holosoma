"""Invariance + KAT tests for the keyed TermSampler (backend-free, Mac-runnable).

These prove the reproducibility contract: a draw is value = f(term, env, episode), independent of the
set/order of other terms, num_envs, and which env subset resets — and the vectorized hash equals a
per-env loop bit-for-bit.
"""

import numpy as np
import pytest
import torch

from holosoma.utils.sampler import STAGE_RESET, STAGE_SETUP, TermSampler, keyed_uniform

SEED = 12345


def _binder(term="randomize_mass_startup", stage=STAGE_RESET, episode=None):
    return TermSampler.bind(SEED, term, stage, episode)


# --- raw hash properties --------------------------------------------------------------------------


def test_keyed_uniform_vectorized_equals_loop():
    envs = np.arange(8)
    vec = keyed_uniform(SEED, 99, 1, envs, 3, 0, 0)
    loop = torch.stack([keyed_uniform(SEED, 99, 1, int(e), 3, 0, 0) for e in range(8)])  # 0-d -> [8]
    assert torch.equal(vec, loop)


def test_keyed_uniform_in_unit_interval():
    u = keyed_uniform(SEED, 7, 1, np.arange(100000), 0, 0, 0)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean().item() - 0.5) < 0.01


def test_keyed_uniform_kat():
    # Pinned wire-format vector: changing the IV/mixer breaks this on purpose.
    assert keyed_uniform(SEED, 99, 1, 7, 3, 0, 0).item() == pytest.approx(0.08690584629403264, abs=1e-12)


# --- TermSampler.draw invariances (the whole point) ----------------------------------------------


def _draw_env(sampler, env_id, leaf=(2.0, 3.0)):
    return sampler.draw(leaf, env_ids=torch.tensor([env_id])).item()


def test_subset_reset_invariance():
    s = _binder()
    all_envs = s.draw((2.0, 3.0), env_ids=torch.arange(8))
    subset = s.draw((2.0, 3.0), env_ids=torch.tensor([2, 5]))
    assert subset[0].item() == all_envs[2].item()
    assert subset[1].item() == all_envs[5].item()


def test_num_envs_invariance():
    s = _binder()
    big = s.draw((2.0, 3.0), env_ids=torch.arange(4096))
    small = s.draw((2.0, 3.0), env_ids=torch.arange(1024))
    assert torch.equal(big[:1024], small)


def _shared_term_draw_within_run(other_terms, episode):
    """Simulate ONE DR-manager run: draw the SHARED term plus a set of OTHER terms (whatever else the
    config happens to include), and return only the shared term's per-(env, episode) values.

    The keyed sampler is stateless (value = f(term, env, episode)), so binding/drawing the other
    terms — in any order, with any params — must NOT perturb the shared term. Drawing them here
    proves that end-to-end, the way a manager would interleave term draws in a real run.
    """
    env_ids = torch.arange(16)
    # The other terms run (different bands, different order between runs) — their draws are discarded;
    # the point is that they exist and execute alongside the shared term.
    for name, band in other_terms:
        TermSampler.bind(SEED, name, STAGE_RESET, episode).draw(
            band, env_ids=env_ids, coords=(torch.arange(4)[None, :],)
        )
    shared = TermSampler.bind(SEED, "shared_term", STAGE_RESET, episode)
    return shared.draw((0.0, 1.0), env_ids=env_ids)


def test_shared_term_invariant_across_different_runs():
    """User-requested consistency guarantee: run DR with one set of terms+params, then AGAIN with
    DIFFERENT terms and DIFFERENT params for everything EXCEPT one shared term — that shared term must
    produce identical values per (env, episode) across both runs. (Independence from the term
    set/order and from other terms' parameters is the whole reason for counter-based keying.)"""
    episode = torch.zeros(16, dtype=torch.long)
    run1 = _shared_term_draw_within_run(
        [("term_A", (0.0, 1.0)), ("term_B", (-2.0, 2.0)), ("term_C", (5.0, 6.0))], episode
    )
    run2 = _shared_term_draw_within_run(
        # Different term SET (A dropped, D/E added), different ORDER, different PARAMS on the rest.
        [("term_E", (10.0, 20.0)), ("term_B", (0.1, 0.2)), ("term_D", (-9.0, -8.0))],
        episode,
    )
    assert torch.equal(run1, run2), "shared term's draws changed when other terms/params changed"
    # And distinct per episode (so it's a real per-(env, episode) draw, not a constant).
    run3 = _shared_term_draw_within_run([("term_B", (0.1, 0.2))], torch.full((16,), 7, dtype=torch.long))
    assert not torch.equal(run1, run3), "shared term did not vary with episode"


def test_term_independence():
    a = TermSampler.bind(SEED, "term_A", STAGE_RESET).draw((0.0, 1.0), env_ids=torch.arange(64))
    b = TermSampler.bind(SEED, "term_B", STAGE_RESET).draw((0.0, 1.0), env_ids=torch.arange(64))
    assert not torch.equal(a, b)


def test_stage_independence():
    setup = TermSampler.bind(SEED, "term_A", STAGE_SETUP).draw((0.0, 1.0), env_ids=torch.arange(64))
    reset = TermSampler.bind(SEED, "term_A", STAGE_RESET).draw((0.0, 1.0), env_ids=torch.arange(64))
    assert not torch.equal(setup, reset)


def test_stream_coord_independence():
    # An int coord is a STREAM tag: two draws differing only in their int coord must decorrelate
    # (this is what used to be ``axis=``). It adds no dimension, so the shape stays [E].
    s = _binder()
    x = s.draw((0.0, 1.0), env_ids=torch.arange(64), coords=(0,))
    y = s.draw((0.0, 1.0), env_ids=torch.arange(64), coords=(1,))
    assert tuple(x.shape) == (64,) and tuple(y.shape) == (64,)
    assert not torch.equal(x, y)


def test_episode_progression_reproducible_and_distinct():
    ep0 = torch.zeros(8, dtype=torch.long)
    ep1 = torch.ones(8, dtype=torch.long)
    s0 = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep0)
    s1 = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep1)
    v0 = s0.draw((0.0, 1.0), env_ids=torch.arange(8))
    v1 = s1.draw((0.0, 1.0), env_ids=torch.arange(8))
    assert not torch.equal(v0, v1)  # different episode -> different value
    # reproducible: same episode again matches
    assert torch.equal(
        v0, TermSampler.bind(SEED, "term_A", STAGE_RESET, ep0.clone()).draw((0.0, 1.0), env_ids=torch.arange(8))
    )


def test_async_episode_per_env():
    # env 5 at episode 12 must match whether or not its neighbours share that episode.
    ep_async = torch.tensor([3, 3, 3, 3, 3, 12, 3, 3], dtype=torch.long)
    ep_all12 = torch.full((8,), 12, dtype=torch.long)
    a = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep_async).draw((0.0, 1.0), env_ids=torch.arange(8))
    b = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep_all12).draw((0.0, 1.0), env_ids=torch.arange(8))
    assert a[5].item() == b[5].item()  # env 5 @ ep12 identical regardless of neighbours
    assert a[0].item() != b[0].item()  # env 0 differs (ep3 vs ep12)


def test_seed_sensitivity():
    a = TermSampler.bind(1, "term_A", STAGE_RESET).draw((0.0, 1.0), env_ids=torch.arange(64))
    b = TermSampler.bind(2, "term_A", STAGE_RESET).draw((0.0, 1.0), env_ids=torch.arange(64))
    assert not torch.equal(a, b)


def test_unseeded_bind_raises():
    # A seed is REQUIRED — there is NO global-RNG fallback. bind(None) must fail loudly so a seedless
    # DR run is caught at bind time rather than silently producing non-reproducible draws.
    with pytest.raises(ValueError, match="requires a base_seed"):
        TermSampler.bind(None, "term_A", STAGE_RESET)


# --- draw shaping reuses the converters ----------------------------------------------------------


def test_draw_gaussian_truncated_in_band():
    s = _binder()
    x = s.draw({"kind": "gaussian", "low": -1.0, "high": 1.0}, env_ids=torch.arange(200000))
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert abs(x.mean().item()) < 0.02


def test_per_actor_loop_draw_reproducible():
    # A per-actor loop pre-draws the full [n_env, 1] vector once and indexes it.
    s = _binder()
    a = s.draw((2.0, 3.0), env_ids=torch.arange(5)).squeeze(-1)
    b = s.draw((2.0, 3.0), env_ids=torch.arange(5)).squeeze(-1)
    assert torch.equal(a, b)
    assert a.min() >= 2.0 and a.max() <= 3.0


def test_draw_int_in_range_and_reproducible():
    s = _binder()
    a = s.draw_int(0, 3, env_ids=torch.arange(1000))
    assert int(a.min()) >= 0 and int(a.max()) <= 3
    b = s.draw_int(0, 3, env_ids=torch.arange(1000))
    assert torch.equal(a, b)


def test_entity_broadcast_shape():
    # A tensor coord adds a trailing dimension: a [1, 3] coord (entity ids on a leading size-1 env
    # axis) broadcasts against the [E]-on-axis-0 env coordinate to [E, 3].
    s = _binder()
    x = s.draw((0.0, 1.0), env_ids=torch.arange(4), coords=(torch.arange(3)[None, :],))
    assert tuple(x.shape) == (4, 3)
    # each entity column independent
    assert not torch.equal(x[:, 0], x[:, 1])


def test_stable_entity_ids_keep_draw_under_reordering():
    # Drawing for entities [10, 20, 30] then [30, 10] must agree per id, not per position — the value
    # keys on the id VALUE, never its position in the coord tensor.
    s = _binder()
    a = s.draw((0.0, 1.0), env_ids=torch.arange(4), coords=(torch.tensor([10, 20, 30])[None, :],))
    b = s.draw((0.0, 1.0), env_ids=torch.arange(4), coords=(torch.tensor([30, 10])[None, :],))
    assert torch.equal(a[:, 0], b[:, 1])  # id 10
    assert torch.equal(a[:, 2], b[:, 0])  # id 30


def test_permute_is_keyed_valid_and_coord_independent():
    # The IsaacSim material bucket-shuffle relies on permute() being a REPRODUCIBLE permutation
    # (so seeded material values are stable run-to-run) with per-channel independence via a stream
    # coord (so the three friction/restitution channels are not rank-aligned).
    s1, s2 = _binder(), _binder()
    p = s1.permute(64, (0,))
    assert sorted(p.tolist()) == list(range(64))  # a genuine permutation
    assert torch.equal(p, s2.permute(64, (0,)))  # same key -> same perm (reproducible)
    assert not torch.equal(p, s1.permute(64, (1,)))  # distinct stream coords -> distinct perms
    assert not torch.equal(p, _binder(term="other_term").permute(64, (0,)))  # term-keyed
    # permute is env/episode-INDEPENDENT (a shared table), unlike draw_int.
    assert torch.equal(
        _binder(episode=torch.zeros(4, dtype=torch.long)).permute(8, (0,)),
        _binder(episode=torch.full((4,), 9, dtype=torch.long)).permute(8, (0,)),
    )


# --- variadic-coordinate shaping: rank comes from the tensor coords' broadcast -------------------


def test_zero_coord_global_flag_shape():
    # A global flag (a single value shared by every env): no caller coords -> output is just the env
    # dimension [E]. This is the degenerate "one stream, per env" draw.
    s = _binder()
    x = s.draw((2.0, 3.0), env_ids=torch.arange(8))
    assert tuple(x.shape) == (8,)
    assert x.min() >= 2.0 and x.max() <= 3.0


def test_geom_pair_2d_entity_shape():
    # A per-(env, i, j) geom-PAIR draw: two tensor coords placed on distinct trailing axes broadcast
    # against the [E]-on-axis-0 env coord -> [E, I, J]. (e.g. a friction-pair matrix between shapes.)
    s = _binder()
    rows = torch.arange(3).reshape(1, 3, 1)  # i on axis 1
    cols = torch.arange(4).reshape(1, 1, 4)  # j on axis 2
    x = s.draw((0.0, 1.0), env_ids=torch.arange(5), coords=(rows, cols))
    assert tuple(x.shape) == (5, 3, 4)
    # distinct (i, j) cells decorrelate
    assert not torch.equal(x[:, 0, 0], x[:, 1, 2])


def test_matrix_draw_shape():
    # A per-(env, r, c) matrix draw (e.g. a 3x3 inertia-like block): same mechanism as the geom-pair,
    # asserting the full [E, R, C] grid materialises with independent cells.
    s = _binder()
    r = torch.arange(3).reshape(1, 3, 1)
    c = torch.arange(3).reshape(1, 1, 3)
    x = s.draw((0.0, 1.0), env_ids=torch.arange(2), coords=(r, c))
    assert tuple(x.shape) == (2, 3, 3)
    flat = x.reshape(2, 9)
    assert flat.unique().numel() == flat.numel()  # all 18 cells distinct (no accidental aliasing)


# --- env and episode are ONE zipped dimension (not an outer product) -----------------------------


def test_env_episode_zipped_single_dimension():
    # env_ids and the per-env episode counter are zipped elementwise into ONE row dimension: row k
    # keys on (env_ids[k], episode[env_ids[k]]) TOGETHER. Changing one env's episode must move only
    # that row, and the output length is len(env_ids) (NOT len(env)*len(episode)).
    ep = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    base = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep).draw((0.0, 1.0), env_ids=torch.arange(4))
    assert tuple(base.shape) == (4,)
    ep2 = ep.clone()
    ep2[2] = 7  # bump only env 2's episode
    bumped = TermSampler.bind(SEED, "term_A", STAGE_RESET, ep2).draw((0.0, 1.0), env_ids=torch.arange(4))
    assert bumped[2].item() != base[2].item()  # env 2's row moved
    assert torch.equal(bumped[[0, 1, 3]], base[[0, 1, 3]])  # the others are untouched (zip, not grid)


def test_env_episode_mismatched_length_raises():
    # The episode tensor must cover every drawn env id (env and episode are the same population, zipped
    # into one dimension). A counter shorter than the referenced env ids is a configuration error.
    short_ep = torch.zeros(4, dtype=torch.long)
    s = TermSampler.bind(SEED, "term_A", STAGE_RESET, short_ep)
    with pytest.raises(IndexError):
        s.draw((0.0, 1.0), env_ids=torch.arange(8))  # env id 7 has no episode entry


# --- the SAME fundamental coordinate draws the SAME value however it is passed --------------------


def test_coord_form_invariance_scalar_vs_tensor():
    """value = f(coordinate VALUES): the SHAPE of a coord only places the value, never changes it.

    So entity id 5 keyed as a scalar int coord, as a 1-element tensor coord, or as one element of a
    multi-element tensor coord must all draw the SAME number for the same (term, env). This is the
    property that lets a per-actor loop (scalar coord per actor) and a vectorized draw (tensor coord
    over all actors) agree bit-for-bit.
    """
    s = _binder()
    env_ids = torch.arange(6)

    # (a) scalar int coord 5  vs  a 1-element tensor coord [5] (on the trailing axis).
    as_int = s.draw((0.0, 1.0), env_ids=env_ids, coords=(5,))  # [E]
    as_len1 = s.draw((0.0, 1.0), env_ids=env_ids, coords=(torch.tensor([5])[None, :],))  # [E, 1]
    assert torch.equal(as_int, as_len1[:, 0])

    # (b) the same id inside a MULTI-element tensor coord lands the identical value at its position.
    multi = s.draw((0.0, 1.0), env_ids=env_ids, coords=(torch.tensor([3, 5, 9])[None, :],))
    assert torch.equal(as_int, multi[:, 1])  # column for id 5

    # (c) per-actor scalar loop == one vectorized tensor-coord draw, element-for-element.
    ids = [10, 20, 30]
    looped = torch.stack([s.draw((0.0, 1.0), env_ids=env_ids, coords=(i,)) for i in ids], dim=-1)
    vectorized = s.draw((0.0, 1.0), env_ids=env_ids, coords=(torch.tensor(ids)[None, :],))
    assert torch.equal(looped, vectorized)

    # (d) a multi-coord draw equals the matching scalar-stream draw cell-by-cell: stream coord k +
    #     entity id e (tensor) must equal the scalar draw coords=(k, e).
    streamed = s.draw((0.0, 1.0), env_ids=env_ids, coords=(2, torch.tensor([7, 8])[None, :]))  # [E, 2]
    cell_70 = s.draw((0.0, 1.0), env_ids=env_ids, coords=(2, 7))  # [E]
    cell_81 = s.draw((0.0, 1.0), env_ids=env_ids, coords=(2, 8))
    assert torch.equal(streamed[:, 0], cell_70)
    assert torch.equal(streamed[:, 1], cell_81)


def test_coord_form_invariance_numpy_python_tensor_equal():
    # The coord container is irrelevant — a Python int, a numpy int, and a 0-d/CPU/CUDA tensor of the
    # same value all key identically (the value is converted to int64 before hashing).
    import numpy as np

    s = _binder()
    env_ids = torch.arange(4)
    base = s.draw((0.0, 1.0), env_ids=env_ids, coords=(5,))
    assert torch.equal(base, s.draw((0.0, 1.0), env_ids=env_ids, coords=(np.int64(5),)))
    assert torch.equal(base, s.draw((0.0, 1.0), env_ids=env_ids, coords=(torch.tensor(5),)))
