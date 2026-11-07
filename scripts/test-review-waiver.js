#!/usr/bin/env node
/**
 * Tests for `scripts/review-waiver.js` — run with
 * `node --test scripts/test-review-waiver.js`.
 *
 * Fixtures are plain JSON under `scripts/fixtures/review-gate/waiver/` so each
 * decision can be replayed from a recorded comment set without the network.
 *
 * `legacyEvaluate` is kept here as the rule the workflow shipped before this
 * change, transcribed from `.github/actions/verify-review-label/action.yml`:
 * label presence, a head-update timestamp read from `created_at` on `committed`
 * and `head_ref_force_pushed` timeline events, and the *label applier's*
 * repository role. Ordinary `committed` events do not carry `created_at`, which
 * is why the staleness branch never fired and a waiver survived any push while
 * the label stayed applied.
 */

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const FIXTURES = path.join(__dirname, "fixtures", "review-gate", "waiver");
const waiver = require("./review-waiver.js");

/** Repository roles as `getCollaboratorPermissionLevel` would report them. */
const ROLES = {
  maint: "maintain",
  writer: "write",
  author: "read",
  someone: "read",
};

function load(name) {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES, name + ".json"), "utf8"));
}

/** The pre-fix decision, transcribed from the shipped composite action. */
function legacyEvaluate(fixture) {
  const events = fixture.timeline || [];
  const labelEvents = events.filter(
    (event) => event.event === "labeled" && event.label && event.label.name
  );
  if (labelEvents.length === 0) return { allowed: false, code: "no-label-event" };
  const labelEvent = labelEvents[labelEvents.length - 1];
  if (fixture.action === "synchronize") return { allowed: false, code: "synchronize" };
  const labelTime = Date.parse(labelEvent.created_at);
  if (!Number.isFinite(labelTime)) return { allowed: false, code: "no-label-time" };
  const headTimes = events
    .filter(
      (event) =>
        ["committed", "head_ref_force_pushed"].includes(event.event) && event.created_at
    )
    .map((event) => Date.parse(event.created_at))
    .filter((time) => Number.isFinite(time));
  const latest = headTimes.length ? Math.max(...headTimes) : null;
  if (latest !== null && labelTime <= latest) {
    return { allowed: false, code: "label-older-than-head" };
  }
  if (!fixture.labelPresent) return { allowed: false, code: "no-label" };
  const actorRole = ROLES[labelEvent.actor && labelEvent.actor.login] || "read";
  if (!["admin", "maintain"].includes(actorRole)) {
    return { allowed: false, code: "label-actor-not-maintainer" };
  }
  return { allowed: true, code: "legacy-label-trusted" };
}

/** Every fixture, with the decision it records in its own `expect` block. */
const FIXTURE_NAMES = fs
  .readdirSync(FIXTURES)
  .filter((name) => name.endsWith(".json"))
  .map((name) => name.slice(0, -".json".length))
  .sort();

function comment(body, extra = {}) {
  return {
    id: 1,
    user: { login: "maint" },
    author_association: "MEMBER",
    created_at: "2026-01-01T00:00:00Z",
    body,
    ...extra,
  };
}

test("the fixture table is complete and every fixture matches its expectation", () => {
  assert.deepEqual(FIXTURE_NAMES, [
    "abbreviated",
    "label-only",
    "match",
    "newest-wins",
    "non-maintainer",
    "permission-maintain",
    "permission-write",
    "reopen-same-sha",
    "stale",
  ]);

  for (const name of FIXTURE_NAMES) {
    const fixture = load(name);
    const decision = waiver.evaluate(fixture);
    assert.equal(
      decision.allowed,
      fixture.expect.allowed,
      `${name}: ${decision.code} — ${decision.reason}`
    );
    assert.equal(decision.code, fixture.expect.code, name);
  }
});

test("the pre-fix rule accepted a waiver for a head that had already moved", () => {
  // The defect meta#89 records: `committed` events carry no `created_at`, so
  // `latestHeadUpdate` was always null and the staleness branch never ran.
  const fixture = load("stale");

  assert.equal(legacyEvaluate(fixture).allowed, true);
  const decision = waiver.evaluate(fixture);
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
  assert.match(decision.reason, /waive-review/);
  assert.match(decision.reason, new RegExp(fixture.headSha));
});

test("a waiver naming the current head is accepted", () => {
  const fixture = load("match");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(decision.code, "waiver-matches-head");
  assert.equal(decision.namedSha, fixture.headSha.toLowerCase());
  assert.equal(decision.evidence.author, "maint");
});

test("an upper-case head and an upper-case waiver still match", () => {
  const sha = "ABCDEF0123456789abcdef0123456789ABCDEF01";
  const decision = waiver.evaluate({
    headSha: sha,
    comments: [comment(`waive-review ${sha.toLowerCase()}`)],
  });

  assert.equal(decision.allowed, true, decision.reason);
});

test("reopening the PR with the same head keeps the waiver valid", () => {
  const fixture = load("reopen-same-sha");
  const decision = waiver.evaluate(fixture);

  assert.equal(fixture.action, "reopened");
  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(decision.code, "waiver-matches-head");
});

test("closing and reopening cannot revive a waiver for another head", () => {
  const fixture = load("reopen-same-sha");
  const decision = waiver.evaluate({
    ...fixture,
    action: "closed",
    headSha: "9".repeat(40),
  });

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
});

test("a comment below the maintainer threshold is rejected", () => {
  const fixture = load("non-maintainer");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
  assert.match(decision.reason, /CONTRIBUTOR/);
});

test("a `write` repository role is below the waiver threshold", () => {
  const fixture = load("permission-write");
  const decision = waiver.evaluate(fixture);

  // Narrower than write on purpose: widening who may waive is a policy change,
  // not part of binding the waiver to a commit.
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
  assert.deepEqual([...waiver.MAINTAINER_PERMISSIONS], ["admin", "maintain"]);
});

test("a `maintain` repository role is accepted even when the association is NONE", () => {
  const decision = waiver.evaluate(load("permission-maintain"));

  assert.equal(decision.allowed, true, decision.reason);
});

test("the label alone grants nothing", () => {
  const fixture = load("label-only");

  assert.equal(legacyEvaluate(fixture).allowed, true);
  const decision = waiver.evaluate(fixture);
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "no-waiver");
  assert.match(decision.reason, /waive-review/);
  assert.match(decision.reason, /label grants nothing/);
});

test("an abbreviated sha is not a waiver", () => {
  const decision = waiver.evaluate(load("abbreviated"));

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "shorthand-waiver");
  assert.equal(decision.namedSha.length, 12);
});

test("the newest maintainer waiver wins over an older one", () => {
  const decision = waiver.evaluate(load("newest-wins"));

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
});

test("a missing or malformed head sha fails closed", () => {
  const fixture = load("match");

  for (const headSha of [undefined, null, "", "not-a-sha", fixture.headSha.slice(0, 12)]) {
    const decision = waiver.evaluate({ ...fixture, headSha });
    assert.equal(decision.allowed, false, `head ${headSha}`);
    assert.equal(decision.code, "invalid-head");
  }
});

test("the bare word does not grant a waiver", () => {
  const decision = waiver.evaluate({
    headSha: "a".repeat(40),
    comments: [comment("I would waive-review this, but let me look first")],
  });

  assert.equal(decision.code, "no-waiver");
});

test("a hex run that is not exactly 40 characters is rejected", () => {
  const headSha = "a".repeat(40);

  for (const named of ["a".repeat(7), "a".repeat(12), "a".repeat(39), "a".repeat(41), "a".repeat(64)]) {
    const decision = waiver.evaluate({
      headSha,
      comments: [comment("waive-review " + named)],
    });

    assert.equal(decision.allowed, false, `length ${named.length}`);
    assert.equal(decision.code, "shorthand-waiver", `length ${named.length}`);
  }
});

test("`waive-review <sha>` survives the forms a maintainer actually types", () => {
  const sha = "b".repeat(40);

  for (const body of [
    `waive-review ${sha}`,
    `waive-review: ${sha}`,
    `waive-review \`${sha}\``,
    `LGTM\n\nwaive-review ${sha}.`,
  ]) {
    const decision = waiver.evaluate({
      headSha: sha,
      comments: [comment(body, { author_association: "OWNER" })],
    });
    assert.equal(decision.allowed, true, body);
  }
});

test("the decision never reads a timeline or committer date", () => {
  const fixture = load("stale");

  // Rewriting every timestamp must not change the verdict: freshness is a
  // SHA comparison, not a clock comparison.
  const rewritten = {
    ...fixture,
    comments: fixture.comments.map((c) => ({ ...c, created_at: "2099-12-31T23:59:59Z" })),
    timeline: [],
  };

  assert.equal(waiver.evaluate(rewritten).code, "stale-waiver");
});

test("the github-script glue fails the job when the waiver is stale", async () => {
  const fixture = load("stale");
  const failures = [];
  const core = { setFailed: (message) => failures.push(message), warning: () => {}, info: () => {} };
  const github = {
    paginate: async () => fixture.comments,
    rest: {
      issues: { listComments: async () => ({ data: fixture.comments }) },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => ({
          data: { role_name: ROLES[username] || "read" },
        }),
      },
    },
  };
  const context = {
    repo: { owner: "tensor4all", repo: "tensor4all-rs" },
    payload: {
      action: fixture.action,
      pull_request: {
        number: 566,
        head: { sha: fixture.headSha },
        labels: [{ name: "rules-review:waive" }],
      },
    },
  };

  const decision = await waiver.run({ github, core, context, labelName: "rules-review:waive" });

  assert.equal(decision.allowed, false);
  assert.equal(failures.length, 1);
  assert.match(failures[0], /waive-review/);
  assert.match(failures[0], /rules-review:waive/);
});

test("the github-script glue passes the job when the waiver matches", async () => {
  const fixture = load("match");
  const failures = [];
  const core = { setFailed: (message) => failures.push(message), warning: () => {}, info: () => {} };
  const github = {
    paginate: async () => fixture.comments,
    rest: {
      issues: { listComments: async () => ({ data: fixture.comments }) },
      repos: {
        getCollaboratorPermissionLevel: async () => ({ data: { role_name: "read" } }),
      },
    },
  };
  const context = {
    repo: { owner: "tensor4all", repo: "tensor4all-rs" },
    payload: {
      action: "labeled",
      pull_request: {
        number: 566,
        head: { sha: fixture.headSha },
        labels: [{ name: "rules-review:waive" }],
      },
    },
  };

  const decision = await waiver.run({ github, core, context, labelName: "rules-review:waive" });

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(failures.length, 0);
});

test("a failed role lookup does not silently authorize a comment", async () => {
  const fixture = load("match");
  const decision = waiver.evaluate({
    headSha: fixture.headSha,
    // No `author_association` and no `author_permission`: the lookup failed.
    comments: [{ ...fixture.comments[0], author_association: "NONE" }],
  });

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
});
