# PRODUCT.md — a prediction market where the resolution can't be rigged

**A prediction market where no human decides the outcome after you place your bet — an AI judge reads only cryptographically-signed evidence, and if it judges wrong, anyone can prove it on-chain and take the liar's money.**

## The problem with prediction markets today

In March 2025, Polymarket ran a market: would Ukraine sign a rare-earth mineral deal with the Trump administration by March 31?

No deal was signed. The market should have resolved NO.

Instead, one large token holder, voting across three accounts, controlled about 25% of the dispute round and forced the answer to YES. Roughly $7 million settled the wrong way. Polymarket said this "wasn't a market failure," so there were no refunds.

Here is the actual flaw, and it is not specific to Polymarket. On these platforms, the question "did it really happen?" gets decided *after* the money is already on the table, by people who hold positions. Resolution becomes a contest of power — whoever controls the most tokens or the most committee seats controls the outcome — instead of a question of fact. The bet you make is not the bet that gets settled.

## How it works

**1. Create — with the rules locked in first.** When someone opens a market, they commit, in advance, everything about how it resolves: the exact question, what counts as proof, which publishers are trusted (by their signing keys), the deadline, and the AI judge that reads the evidence. All of this is fixed and public *before anyone can trade.* You see the full resolution procedure from the first dollar.

**2. Trade.** People buy and sell YES/NO shares against an automated market maker. Because the resolution rules were committed up front, you are pricing the procedure, not gambling on who shows up to vote later.

**3. Evidence — only signed proof gets in.** After the deadline, anyone can submit evidence. But you cannot submit a paragraph of opinion. A submission is admitted *only* if it carries a real cryptographic signature from a pinned publisher — a signed news image (C2PA), a DKIM-signed Reuters or AP alert, a signed price feed. The system checks the signature on-chain, natively, bit-for-bit against the publisher's real key. You can bring any genuinely-published item. You cannot bring an invented one.

**4. The AI judge resolves — and being wrong is provable.** A fixed, public AI judge reads the admitted evidence and outputs the verdict. It runs deterministically: the same evidence always produces the same answer. If the resolver lies about what the judge said, anyone can recompute it, challenge, and force a dispute that narrows down to a single arithmetic step the Sui blockchain itself re-runs to settle who is right. The liar is slashed. The honest verdict stands.

## Why you can trust it

**No human decides anything after trading opens.** Every judgment call — the question, the trusted sources, the decision rule — is fixed at creation, when no money is at stake yet. After that, resolution is a pure calculation over signed bytes. There is no settlement-time vote to capture. The exact thing that broke the Ukraine market does not exist here.

**The AI judge is the real model, not a watered-down stand-in.** The judge is Qwen3-0.6B running in a deterministic virtual machine. We measured its quality: perplexity 34.5974, matching the published floating-point reference (34.60) and the standard llama.cpp build (34.99). It is the model as shipped. There is nothing dumbed down to make the math cheaper.

**The judge runs identically everywhere — provably.** The judge produces bit-for-bit identical output regardless of how many CPU threads it uses, whether it runs on an Apple M4 GPU or a CPU, or on a different machine entirely. "Run it yourself and check" is a real option, not a slogan.

**A wrong verdict is provable on-chain, and we already proved one.** We injected a real fault into the model and convicted it: the dispute narrowed a 29.5-million-step judgment down to one micro-operation, and a Sui Move contract of a few hundred lines verified the fault and slashed the cheater. The test (`dispute/tests/qwen_conviction.move`) passes today. Worst case, a full dispute takes about 38 transactions, each at the chain's minimum gas. One honest person with a laptop and a bond makes every wrong verdict unprofitable — and the people most motivated to check are the traders already in the market.

**Honesty is free.** On the honest path, the proof machinery adds roughly 0% overhead. Being verifiable costs nothing when no one is cheating.

One honest boundary, stated plainly: we prove the committed model ran correctly on genuinely-signed evidence. We do not claim the model's judgment is always *wise* — that is the same trust you place in any named judge, except this judge is fixed, public, and provably executed. And a publisher can still sign a falsehood under its own key, which is the same trust a byline already carries, now made explicit and auditable. The decision rule defends against this: a YES needs agreement from several *independent* trusted sources, not one, and confirmations are counted by trust root so that one wire story republished by 40 outlets still counts once. When sources genuinely conflict or evidence is too thin, the market resolves UNRESOLVED and refunds rather than guessing — because a wrong answer is worse than no answer.

## What you can build on it

This is an oracle with no oracle inside it: a way to turn the world's already-signed facts into an on-chain answer that no single party has to be trusted for. Prediction markets are just the first thing you build on it.

- **Prediction markets** that cannot be captured at settlement.
- **Parametric insurance** — flight delays, weather, outages — that pays out automatically from signed feeds, with no claims adjuster to argue with.
- **Oracles** for any signed real-world fact: prices, election results, official filings, published reports.
- **Bridges and settlement** that depend on real-world conditions being provably true.
- **Verifiable agents and KYC gates** that act only on cryptographically-attested inputs.
- **A computation graph** — one market's proven verdict becomes a signed input to the next, because the chain itself signs the answer.

The world already signs its facts. This is the machine that computes over those signatures and makes the answer something nobody has to be trusted for.

*Deeper reading: [VISION.md](VISION.md) (the general system), [EVIDENCE.md](EVIDENCE.md) (how evidence is chosen and the judge is hardened), [DEMO.md](DEMO.md) (the working demo).*
