"""
Odds calculator — American odds utility.
Usage:
    python odds.py 410
    python odds.py -115
    python odds.py 410 -115   (compare multiple lines)
"""

import sys


def implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def fair_odds(prob: float) -> str:
    """Convert probability back to American odds."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob < 0.5:
        val = (100 / prob) - 100
        return f"+{val:.0f}"
    else:
        val = prob / (1 - prob) * 100
        return f"-{val:.0f}"


def to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def to_win(odds: int, stake: float = 100) -> float:
    if odds > 0:
        return stake * odds / 100
    return stake * 100 / abs(odds)


def analyze(odds: int):
    prob    = implied_prob(odds)
    decimal = to_decimal(odds)
    profit  = to_win(odds)

    sign = "+" if odds > 0 else ""
    print(f"\n  Odds          : {sign}{odds}")
    print(f"  Implied prob  : {prob*100:.1f}%")
    print(f"  Decimal       : {decimal:.3f}")
    print(f"  Profit on $100: ${profit:.2f}")


def compare(lines: list[int]):
    print(f"\n  {'Odds':<10} {'Impl%':>8} {'Decimal':>9} {'Profit/100':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*12}")
    for odds in lines:
        prob    = implied_prob(odds)
        decimal = to_decimal(odds)
        profit  = to_win(odds)
        sign    = "+" if odds > 0 else ""
        print(f"  {sign}{odds:<9} {prob*100:>7.1f}% {decimal:>9.3f} {profit:>11.2f}")

    total_prob = sum(implied_prob(o) for o in lines)
    print(f"\n  Total implied : {total_prob*100:.1f}%  (vig: {(total_prob-1)*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python odds.py <odds> [odds2 ...]")
        print("Examples:")
        print("  python odds.py 410")
        print("  python odds.py -115 -105   (over/under comparison)")
        sys.exit(1)

    lines = [int(a) for a in sys.argv[1:]]

    if len(lines) == 1:
        analyze(lines[0])
    else:
        compare(lines)
