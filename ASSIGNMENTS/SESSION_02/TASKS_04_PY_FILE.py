from datetime import date, datetime

# Known ordered labels commonly seen in Flipkart order data
ORDERED_SETS = [
    ["placed", "packed", "shipped", "out for delivery", "delivered"],
    ["low", "medium", "high"],
    ["s", "m", "l", "xl", "xxl"],
    ["poor", "average", "good", "excellent"],
]


def is_date_like(v):
    """True for date/datetime objects or strings like '2026-10-21'."""
    if isinstance(v, (date, datetime)):
        return True
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                datetime.strptime(v.strip(), fmt)
                return True
            except ValueError:
                pass
    return False


def detect_scale(values):
    """Guess the measurement scale of a column: Nominal, Ordinal, Interval or Ratio."""
    data = [v for v in values if v is not None and v != ""]
    if not data:
        return "Unknown (empty column)"

    # Rule 1: dates / timestamps -> Interval (differences are meaningful, no true zero)
    if all(is_date_like(v) for v in data):
        return "Interval"

    # Rule 2: numbers
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data):
        unique = sorted(set(data))
        # few whole-number levels between 1 and 10 (e.g. 1 to 5 star ratings) -> Ordinal
        if all(float(v).is_integer() for v in unique) and len(unique) <= 5 and unique[0] >= 1 and unique[-1] <= 10:
            return "Ordinal"
        # negative values mean the zero point is arbitrary -> Interval
        if min(data) < 0:
            return "Interval"
        # non-negative with a meaningful zero (price, quantity, discount) -> Ratio
        return "Ratio"

    # Rule 3: text
    labels = {str(v).strip().lower() for v in data}
    for ordered in ORDERED_SETS:
        if labels <= set(ordered):
            return "Ordinal"       # labels follow a natural order
    return "Nominal"               # plain labels with no order


if __name__ == "__main__":
    flipkart = {
        "Product Category": ["Mobiles", "Fashion", "Grocery", "Mobiles", "Books"],
        "Payment Method": ["UPI", "COD", "Card", "UPI", "COD"],
        "Order Status": ["Placed", "Shipped", "Delivered", "Packed", "Delivered"],
        "Order Date": ["2026-10-01", "2026-10-05", "2026-10-09", "2026-10-12", "2026-10-20"],
        "Product Rating": [5, 4, 3, 5, 2],
        "Order Amount (Rs)": [499.0, 1299.0, 249.5, 15999.0, 349.0],
        "Quantity": [1, 2, 1, 3, 12],
        "Temperature Change": [-3, 0, 5, 8, -1],
    }
    for name, col in flipkart.items():
        print(f"{name:20} -> {detect_scale(col)}")
