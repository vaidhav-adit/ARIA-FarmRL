# Reference tomato growth schedule used to compare actual plant progress to what is expected each day.


# Dictionary from day-of-season to target cumulative growth score (interpolated between keys).
TOMATO_GROWTH_CURVE = {  # Maps milestone days to normalized expected growth in [0, 1].
    0: 0.02,  # Beginning of season — very small biomass.
    7: 0.05,  # End of first week.
    14: 0.10,  # Two weeks.
    21: 0.20,  # Three weeks.
    28: 0.30,  # Four weeks.
    35: 0.40,  # Mid-vegetative phase.
    42: 0.50,  # About halfway through the curve.
    49: 0.58,  # Approaching reproductive phase.
    56: 0.65,  # Continued maturation.
    63: 0.70,  # Later mid-season.
    70: 0.75,  # Nearing harvest window.
    77: 0.80,  # Strong fruit fill stage.
    84: 0.83,  # Near final target.
    90: 0.85,  # Season cap used for day-90 comparisons.
}


def expected_growth(day: int) -> float:  # Linearly interpolates the table for any integer day in [0, 90].
    day = max(0, min(90, day))  # Clamp day so lookups never go out of documented range.
    days = sorted(TOMATO_GROWTH_CURVE)  # Ordered list of knot days for interpolation.
    for i in range(len(days) - 1):  # Scan each segment [d1, d2].
        d1, d2 = days[i], days[i + 1]  # Segment endpoints.
        if d1 <= day <= d2:  # If current day lies in this segment.
            ratio = (day - d1) / (d2 - d1)  # Fraction between d1 and d2 (0 at d1, 1 at d2).
            return round(  # Return interpolated growth, rounded for stable reward arithmetic.
                TOMATO_GROWTH_CURVE[d1]  # Value at left endpoint.
                + ratio * (TOMATO_GROWTH_CURVE[d2] - TOMATO_GROWTH_CURVE[d1]),  # Linear blend to right endpoint.
                4,  # Four decimal places.
            )
    return TOMATO_GROWTH_CURVE[90]  # Fallback if loop did not match (should be rare after clamp).
