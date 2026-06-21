# Bao la Kiswahili — Rules

> Source: <http://www.gamecabinet.com/rules/Bao.html>  
> Article by Rob Nierse, Waterlelieweg 56, 2215 GP Voorhout, Holland

---

## Table of Contents

1. [The Board and Starting Position](#the-board-and-starting-position)
2. [Goal of the Game](#goal-of-the-game)
3. [Namua Stage](#namua-stage)
   - [Entering from the Stock](#entering-from-the-stock)
   - [You Must Capture If You Can](#you-must-capture-if-you-can)
   - [Entering Captured Seeds](#entering-captured-seeds)
   - [Entering More Than One Seed](#entering-more-than-one-seed)
   - [Kichwa and Kimbi](#kichwa-and-kimbi)
   - [Capturing with Captured Seeds](#capturing-with-captured-seeds)
   - [Multiple Captures](#multiple-captures)
   - [Continuing to Sow](#continuing-to-sow)
   - [Sowing Around the Corner](#sowing-around-the-corner)
   - [Takasa — Moving Without Capture](#takasa--moving-without-capture)
   - [The Nyumba (House)](#the-nyumba-house)
4. [Mtaji Stage](#mtaji-stage)
   - [Capture](#capture)
   - [Takasa in Mtaji](#takasa-in-mtaji)
5. [Summary of Key Rules](#summary-of-key-rules)
6. [Differences from the Mancala World Ruleset](#differences-from-the-mancala-world-ruleset)
7. [Notation](#notation)
8. [Glossary](#glossary)

---

## The Board and Starting Position

A Bao board has four rows of eight holes each. The two rows closest to you are yours; the two rows closest to your opponent are theirs. One hole per player — the **fifth hole from the left on the front row** — is marked as the **nyumba** (house).

**Starting position** (your side on the bottom, opponent on top):

```
0 0 0 0 0 0 0 0   ← opponent back row
0 2 2 6 0 0 0 0   ← opponent front row  (nyumba = hole 4 from left)
0 0 0 0 6 2 2 0   ← your front row      (nyumba = hole 5 from left)
0 0 0 0 0 0 0 0   ← your back row
```

Each player starts with **10 seeds on the board** and **22 seeds in hand** (the stock), for 32 total. Seeds are called **kete**.

The game is divided into two stages:

- **Namua stage** — each turn a player adds one seed from the stock to the board.
- **Mtaji stage** — begins when both stocks are empty; players sow only seeds already on the board.

---

## Goal of the Game

You win by either:

- **Depleting your opponent's entire front row** (no seeds remain in any of their 8 front-row holes), or
- **Depriving your opponent of all legal moves**.

A win during the namua stage (while seeds remain in hand) is called a win **mkononi** ("in hand"). There are no draws, though theoretically endless cycles are possible.

---

## Namua Stage

### Entering from the Stock

On your turn, look at your **front row**. Find a hole that:

1. Contains **one or more seeds**, and
2. Has an **opposing hole** (directly across in your opponent's front row) that also contains **one or more seeds**.

Take one seed from your stock and place it into that hole. Then **capture** all seeds from the opposing hole.

**Three conditions must be met to capture:**

1. Your chosen hole in your front row must already contain seeds.
2. The opposing hole must contain seeds.
3. You place exactly one seed from your stock into your hole.

### You Must Capture If You Can

**Capturing is mandatory.** If any valid capture is available, you must take it. You may not voluntarily pass on a capture.

### Entering Captured Seeds

In Bao, captured seeds are immediately re-entered into play on your side — they are not removed from the game.

Place the captured seed(s) starting from the **extreme left or right hole** of your front row. These end holes are called **kichwa** ("head"). You may choose left or right freely, *unless* the kichwa/kimbi rule applies (see below).

If the first seed lands in an **empty hole**, the move ends.

### Entering More Than One Seed

If you captured multiple seeds, **sow** them one at a time into consecutive holes of your front row, starting at a kichwa (leftmost or rightmost hole). Never skip a hole.

**Example** — capturing 3 seeds and sowing from the left:

```
Before:   0 0 0 0 3 0 0 0   (opponent front row)
          0 0 0 0 7 0 0 0   (your front row)

Place one stock seed in your hole 5 (7 → 8) and capture the opposing 3 seeds:
                 0 0 0 0 0 0 0 0   (opponent hole 5 emptied)
                 0 0 0 0 8 0 0 0   (your hole 5 gains the stock seed)

Sowing 3 captured seeds from left kichwa:
                 1 1 1 0 8 0 0 0   ← last seed in hole 3 (empty → move ends)
```

### Kichwa and Kimbi

You **cannot choose** which side to sow from in these cases:

- **Kichwa** — the extreme left (hole 1) and extreme right (hole 8) of the front row.
- **Kimbi** — the second-from-left (hole 2) and second-from-right (hole 7) of the front row.

If you capture from a **kichwa or kimbi hole**, you must sow the captured seeds starting from the **same side** as that hole.

| Hole captured | Must sow from |
|---|---|
| Hole 1 (left kichwa) | Left |
| Hole 2 (left kimbi) | Left |
| Hole 7 (right kimbi) | Right |
| Hole 8 (right kichwa) | Right |

You must generally **continue in the same direction** for subsequent sowing in that turn. The **only exception** is a mid-turn chain capture that lands on a kichwa or kimbi hole — in that case the re-entry kichwa is forced to the nearest end, which may flip the direction. This is the only way to change sowing direction within a single move.

### Capturing with Captured Seeds

After sowing your captured seeds, check where the **last seed** lands:

- **Last seed lands in an occupied hole with seeds in the opposing hole** → capture those opposing seeds immediately and continue sowing.
- **Last seed lands in an occupied hole with an empty opposing hole** → pick up all seeds from that hole and continue sowing in the same direction (do not capture).
- **Last seed lands in an empty hole** → your turn ends.

Your turn can only end when your last seed falls in an empty hole.

### Multiple Captures

A single turn can produce a chain of captures. Each time the last sown seed lands opposite an occupied hole, you capture and continue. This can result in a dramatic sequence of multiple captures in one turn.

### Continuing to Sow

If the last seed lands in an occupied hole whose **opposing hole is empty**, you cannot capture. Instead, pick up **all seeds from that hole** and continue sowing in the **same direction**, starting with the very next hole. Do not change direction.

This applies even if sowing carries you into your **back row** and back around.

### Sowing Around the Corner

If you run out of holes in the front row while sowing, continue into the back row. If you have enough seeds, you can wrap back around into the front row again.

Hole order when sowing **left**: front row right-to-left, then back row left-to-right (wrapping around the left end).  
Hole order when sowing **right**: front row left-to-right, then back row right-to-left (wrapping around the right end).

### Takasa — Moving Without Capture

If you **cannot** begin a move with a capture (no occupied front-row hole has an occupied opposing hole), you must play **takasa**:

1. Choose any **occupied** hole in your front row (one or more seeds — a singleton is allowed here, see below).
2. Place one seed from your stock into it. A singleton now holds **2 seeds**, so it can be sown.
3. Pick up all seeds from that hole and sow them left or right.
4. **No captures are allowed** during a takasa move, even if the last seed lands opposite an occupied hole.

> A **singleton may be sown as a namua takasa source**: the stock seed you add in step 2 lifts it to 2 seeds before sowing. The "never sow a singleton" rule applies only in the **mtaji stage**, where no stock seed is added (see [Takasa in Mtaji](#takasa-in-mtaji)).

> **You may not enter the seed into a functional nyumba on a takasa move** unless the nyumba is the only occupied hole in your front row. See [The Nyumba](#the-nyumba-house).

---

## The Nyumba (House)

The **nyumba** is the fifth hole from the left on your front row (marked with a rectangle on the physical board). It has special rules as long as it contains **6 or more seeds** (a *functional* nyumba). Once the nyumba's seeds have been sown out, it becomes an ordinary hole.

### You May Not Enter the Nyumba on a Takasa Move

When you play a **takasa** (a move that does not begin with a capture), you may **not** place your stock seed into a functional nyumba — **unless the nyumba is the only occupied hole in your front row** (the case covered by [Takasa with Only the Nyumba Remaining](#takasa-with-only-the-nyumba-remaining)).

This protects the nyumba from being voluntarily broken open: during the namua stage a full nyumba can only be unleashed by sowing landing in it during a **capturing** move, never by a player choosing to empty it with a takasa. Entering the nyumba **to make a capture** is always allowed.

If the nyumba holds fewer than 6 seeds it is an ordinary hole, so this restriction no longer applies.

### Stopping Sowing at the Nyumba

Normally, if your last seed lands in an occupied hole with an empty opposing hole, you must continue sowing. The nyumba is an **exception**: if your last seed lands in the nyumba and the opposing hole is empty, you **may choose** to end your turn and wait.

This is strategically important — a full nyumba can be unleashed at the right moment for a decisive attack.

### Takasa with Only the Nyumba Remaining

If the nyumba is the **only occupied hole** in your front row and you cannot capture, the normal takasa procedure is modified:

- Place one seed from your stock into the nyumba.
- Remove **two seeds** from the nyumba and sow them left or right.

This special rule only applies when the nyumba contains **6 or more seeds**.

---

## Summary of Key Rules

| Rule | Detail |
|---|---|
| **Must capture** | Always capture if a capture is available. |
| **Entering captured seeds** | Always enter from the left or right kichwa (first hole). |
| **Kichwa/kimbi forces direction** | If you captured a kichwa or kimbi, sow from that same side. |
| **Occupied hole, occupied opposite** | Capture the opposing seeds and continue. |
| **Occupied hole, empty opposite** | Pick up seeds from that hole and continue sowing in the same direction. |
| **Empty hole** | Move ends. |
| **No singletons (mtaji)** | In **mtaji** you may never sow a hole containing only 1 seed. In **namua takasa** a singleton may be the source — the stock seed added first makes it 2. |

---

## Mtaji Stage

The mtaji stage begins when both players have played all seeds from their stocks. No new seeds enter the board.

### Capture

You must capture if possible. To do so, find a hole (front or back row) such that sowing its seeds causes the **last seed of the first lap to land in a front-row hole with an occupied opposing hole** (that opposing hole is called the **mtaji**).

Rules:
- You may sow from either the front or back row.
- You may never sow a singleton.
- The last seed **of the first lap** must land in the front row opposite a hole with seeds. (The **first-lap rule**: if the first lap does not end in a capture, the whole move captures nothing — even if a later relay lap happens to land opposite a loaded hole. See [Takasa in Mtaji](#takasa-in-mtaji).)
- Once the first lap captures, the chain of captures/continuations proceeds exactly as in namua.

**Example:**

```
0 0 0 0 5 6 0 0   (opponent front row)
0 3 0 0 4 1 0 0   (your front row)
0 9 0 0 0 0 0 0   (your back row)
```

You can sow the 3-seed hole to the right (last seed lands opposite the 5 → capture), or sow the 9-seed hole to the right around the corner (last seed lands opposite the 6 → capture).

### Takasa in Mtaji

If no capture is possible — i.e. no hole has a **first-lap** capture — play **takasa**: sow any non-singleton hole from your front row left or right. Pick up all of the hole's seeds and sow them; if the last seed lands in an occupied hole, pick that hole up and continue in the same direction (a relay), until the last seed falls in an empty hole. **No captures occur during takasa**, even if a relay lands opposite a loaded hole. If no non-singleton front-row hole exists, you may instead sow any non-singleton hole from your back row.

**Mtaji moja** (last mtaji) rule: if your opponent has only **one** mtaji hole left, you may not sow that hole in a takasa situation — doing so would deprive them of their only target. You must sow a different hole.

---

## Differences from the Mancala World Ruleset

This document is based on the original **Nierse / GameCabinet** article
([source](http://www.gamecabinet.com/rules/Bao.html)). The
[Mancala World (Ralf Gering) ruleset](https://mancala.fandom.com/wiki/Bao_la_Kiswahili)
is another widely-cited write-up of championship Bao, and the two disagree (or
differ in completeness) on the points below. The **GameCabinet (source)** column
records what the original article says, **RULES.md** what this document now states,
and the **Implemented?** column whether this engine encodes the rule.

| # | Rule point | GameCabinet (source) | RULES.md (this document) | Mancala World (Fandom) | Implemented? |
|---|---|---|---|---|---|
| 1 | **Takasa entry into the nyumba** | Silent — no restriction stated; only the "Takasa with Only Your Nyumba Remaining" special rule is given. | Forbidden on a non-capturing move unless the nyumba is the only occupied front hole; capturing entry is always allowed. | Same as RULES.md, explicit: *"he is not permitted to put the seed into it, unless it is the only occupied hole in his front row."* | ✅ Yes — enforced in the namua legal-action mask (`_namua_mask`, `forbid_nyumba_takasa`). |
| 2 | **Nyumba stop on a takasa lap (namua)** | Optional in all cases, conditioned on the opposing hole being empty: *"the player may end his turn if he wishes."* | Always optional ("may choose to end your turn"). | Mandatory: a takata lap ending in the nyumba ends the turn *"without delay"*; only a **capturing** lap offers the free choice (*safari*). | ⚠️ Partial — engine offers the optional stop/continue choice in both cases. |
| 3 | **Nyumba stop in the mtaji stage** | Silent; mtaji "is not very different from the namua stage," implying the same optional stop. | "Same as namua" — implies the optional stop still applies. | The optional stop is namua-only; in mtaji the player **must** *safari* (continue) when a lap ends in the nyumba. | ⚠️ No — engine does not force continuation in mtaji. |
| 4 | **16-seed capture cutoff (mtaji)** | Absent. | Absent. | *"If 16 or more seeds are sown in the first lap, nothing will be captured."* (mtaji only) | ❌ No. |
| 5 | **Singleton sowing in takasa** | Namua takasa allows any hole with *"one or more seeds"* (the stock seed lifts a singleton to 2); only mtaji forbids starting on a singleton (no exception). | Same as GameCabinet: a singleton **may** be a namua-takasa source because the stock seed makes it ≥ 2; singletons may never be sown in mtaji. | Must sow a hole with ≥ 2 seeds *"unless all non-empty holes in the front row are singletons."* | ✅ Yes — namua takasa permits singleton sources (`_namua_mask` requires only `board[0, c] > 0`); mtaji forbids singletons (`_mtaji_capture_mask`/`_mtaji_mask`). The Mancala World *all-singletons* exception is not implemented. |
| 6 | **"Front row may never be emptied" (mtaji)** | Absent. | Absent. | Front row may never be emptied, even temporarily; a sole kichwa with ≥ 2 seeds must be sown toward the centre. | ❌ No. |
| 7 | **Takasia / mtaji-moja** | Has mtaji-moja only: *"the only mtaji left for your opponent ... may not be sown in a takasa situation."* | Simplified: you may not sow a hole that is the opponent's only mtaji. | Fuller *takasia* rule with the "reached from a nyumba" exception; a nyumba itself cannot be takasia'd, nor the only occupied / only multi-seed hole. | ⚠️ Partial — mtaji-moja source block only (`mtaji_moja_active`). |
| 8 | **First-lap rule / capture-chaining** | Implicit: takasa allows no captures, and *"you always keep on sowing or capturing."* | A move captures only if its **first lap** ends in a capture; otherwise it is a takasa and captures nothing, even on a later relay lap. | Stated explicitly: *"If the first lap of a move doesn't capture, nothing will be captured in the full move."* | ✅ Yes — namua takasa runs with `allow_capture=False`; mtaji gates the whole move on `first_lap_is_capture` (`mtaji_step`, `_mtaji_capture_mask`). |

> Differences **#1** (takasa entry into the nyumba), **#5** (singleton
> namua-takasa sources) and **#8** (the mtaji first-lap rule) are enforced by
> the engine. The remaining rows — **#2, #3, #4, #6** and the fuller **#7**
> *takasia* — are documented here for reference and are **not** implemented;
> they remain as the original Nierse ruleset describes them.

---

## Notation

Each hole is numbered 1–8 per row. Rows are named:

```
8 7 6 5 4 3 2 1   b  (opponent back row)
8 7 6 5 4 3 2 1   a  (opponent front row)
1 2 3 4 5 6 7 8   A  (your front row)
1 2 3 4 5 6 7 8   B  (your back row)
```

| Symbol | Meaning |
|---|---|
| `A5R` | Sow from your front-row hole 5, to the right |
| `B3L` | Sow from your back-row hole 3, to the left |
| `>` | The nyumba seeds are now sown (e.g. `A5R>`) |
| `*` | Takasa move (no capture), e.g. `A3R*` |

---

## Glossary

| Term | Definition |
|---|---|
| **Bao** | 1. The game; 2. the board; 3. a goal (scored in soccer) |
| **Bao Hamna** | Victory; clearance of the opponent's front row (*hamna* = "there is not") |
| **kete** | Seed(s) used in Bao, from the Mkomwe tree |
| **kichwa** | The far-end hole on either side of the front row; also means "head" |
| **kimbi** | The second hole from either end of the front row (holes 2 and 7) |
| **mkononi** | "In the hand" — a victory achieved during the namua stage while seeds remain in stock |
| **mtaji** | An occupied hole in the opponent's front row that is opposite the last-sown seed; also the name of the second stage |
| **nyumba** | "House" — the marked hole (5th from the left) in each player's front row |
| **Piga Tanji** | To attack several houses at the same time |
| **singleton** | A hole containing exactly one seed. It may never be sown in **mtaji**; in **namua takasa** it may be the source, since the stock seed added first makes it 2 |
| **takasa / takata** | A move played without capture |
| **takasia** | Playing without capturing in a way that forces the opponent to also play without capturing, setting up a future capture |

---

*Rules compiled by Rob Nierse; primary research by Alex de Voogt (1991–1995) with assistance from Bao masters of Zanzibar.*  
*Primary source: A.J. de Voogt, "Limits of the Mind" (1995), Research School CNWS, Leiden.*
