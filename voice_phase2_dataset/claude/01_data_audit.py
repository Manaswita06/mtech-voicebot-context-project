"""
01_data_audit.py

Run this FIRST, before building anything else, and re-run it if you get a
new export of the dataset. It checks whether the numeric voice_attributes
(emotion_intensity, speaking_rate, prosody) actually carry signal, or are
placeholder/random values. Trusting a noisy field as ground truth is the
single easiest way to lose a viva, so this check is not optional.

Usage:
    python 01_data_audit.py emotional_conversations.json
"""
import json
import sys
import statistics
import collections
from pathlib import Path


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def structural_summary(data):
    fam = collections.Counter(d["scenario_family"] for d in data)
    scen = collections.Counter(d["scenario"] for d in data)
    split = collections.Counter(d["split"] for d in data)
    esc = collections.Counter(d["escalated"] for d in data)
    res = collections.Counter(d["resolution_status"] for d in data)
    turns = [d["num_turns"] for d in data]

    print("=" * 70)
    print("STRUCTURAL SUMMARY")
    print("=" * 70)
    print(f"conversations: {len(data)}")
    print(f"scenario families: {len(fam)}  |  unique scenarios: {len(scen)}")
    print(f"split: {dict(split)}")
    print(f"escalated label variance: {dict(esc)}")
    print(f"resolution_status variance: {dict(res)}")
    print(f"turns per conversation: min={min(turns)} max={max(turns)} "
          f"mean={statistics.mean(turns):.2f}")
    if len(esc) == 1 or len(res) == 1:
        print("\n  WARNING: outcome label has zero variance. Any experiment that\n"
              "  claims to 'predict escalation' or 'predict resolution' on this\n"
              "  file is not measurable. Do not build that experiment on this data.")


def text_reuse(data):
    texts = collections.Counter()
    total = 0
    for d in data:
        for t in d["turns"]:
            texts[t["text"]] += 1
            total += 1
    print("\n" + "=" * 70)
    print("TEXT TEMPLATING")
    print("=" * 70)
    print(f"unique turn texts: {len(texts)} out of {total} total turns "
          f"({len(texts) / total:.1%} unique)")
    print("most repeated lines:")
    for line, n in texts.most_common(5):
        print(f"  {n:4d}x  {line!r}")


def emotion_vocab_by_role(data):
    cust, agent = collections.Counter(), collections.Counter()
    for d in data:
        for t in d["turns"]:
            e = t["voice_attributes"]["emotion"]
            (cust if t["role"] == "customer" else agent)[e] += 1
    print("\n" + "=" * 70)
    print("EMOTION VOCABULARY BY ROLE (this part IS designed, not random)")
    print("=" * 70)
    print(f"customer vocab ({len(cust)}): {dict(cust)}")
    print(f"agent vocab ({len(agent)}): {dict(agent)}")
    overlap = set(cust) & set(agent)
    print(f"overlap between customer and agent vocabularies: {overlap or 'none'}")


def numeric_field_randomness_check(data):
    """
    The core diagnostic. If emotion_intensity / speaking_rate / prosody were
    designed to reflect the emotion label, we'd expect:
      - intensity distribution to differ meaningfully across emotion labels
      - speaking_rate distribution to differ across emotion labels
    If instead every emotion shows roughly the same distribution, the field
    is very likely sampled independently of the label (i.e. decorative).
    """
    print("\n" + "=" * 70)
    print("NUMERIC FIELD SIGNAL CHECK")
    print("=" * 70)

    all_intensity = []
    rate_by_emotion = collections.defaultdict(collections.Counter)
    intensity_by_emotion = collections.defaultdict(list)

    for d in data:
        for t in d["turns"]:
            va = t["voice_attributes"]
            all_intensity.append(va["emotion_intensity"])
            rate_by_emotion[va["emotion"]][va["speaking_rate"]] += 1
            intensity_by_emotion[va["emotion"]].append(va["emotion_intensity"])

    all_intensity.sort()
    n = len(all_intensity)
    print(f"emotion_intensity global range: "
          f"[{all_intensity[0]:.2f}, {all_intensity[-1]:.2f}], "
          f"mean={statistics.mean(all_intensity):.3f}")
    print("decile check (a straight linear ramp here == uniform random draw):")
    for i in range(10):
        print(f"  decile {i}: {all_intensity[int(i * n / 10)]:.3f}")

    print("\nmean emotion_intensity per emotion label "
          "(should differ a lot if intensity is meaningful):")
    means = {e: statistics.mean(v) for e, v in intensity_by_emotion.items()}
    spread = max(means.values()) - min(means.values())
    for e, m in sorted(means.items(), key=lambda x: -x[1]):
        print(f"  {e:12s} mean={m:.3f}  n={len(intensity_by_emotion[e])}")
    print(f"  -> spread across all emotions: {spread:.3f} "
          f"(global stdev is ~{statistics.pstdev(all_intensity):.3f}; "
          f"if spread << stdev, intensity is not separating emotions)")

    print("\nspeaking_rate distribution per emotion "
          "(should differ if rate reflects emotion):")
    for e in list(rate_by_emotion)[:6]:
        c = rate_by_emotion[e]
        tot = sum(c.values())
        norm = {k: round(v / tot, 2) for k, v in c.items()}
        print(f"  {e:12s} n={tot:4d}  {norm}")

    if spread < 0.05:
        print("\n  CONCLUSION: emotion_intensity does not separate emotion labels.\n"
              "  Treat it as noise. Do not use it as a ground-truth acoustic\n"
              "  target for any model. Regenerate real values via TTS + feature\n"
              "  extraction instead (see 02_ and 03_ scripts).")


def speaker_roster(data):
    cust = collections.Counter(d["customer_speaker"] for d in data)
    agent = collections.Counter(d["agent_speaker"] for d in data)
    print("\n" + "=" * 70)
    print("SPEAKER ROSTER (for TTS voice-cloning targets)")
    print("=" * 70)
    print(f"distinct speaker identities used across both roles: "
          f"{len(set(cust) | set(agent))}")
    print(f"as customer: {dict(cust)}")
    print(f"as agent:    {dict(agent)}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/emotional_conversations.json"
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)
    data = load(path)
    structural_summary(data)
    text_reuse(data)
    emotion_vocab_by_role(data)
    numeric_field_randomness_check(data)
    speaker_roster(data)


if __name__ == "__main__":
    main()