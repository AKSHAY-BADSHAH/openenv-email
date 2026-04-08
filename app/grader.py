def grade_easy(response):
    response = response.lower()
    if "spam" in response:
        return 1.0
    elif "promotion" in response:
        return 0.7
    elif "important" in response:
        return 0.3
    return 0.0


def grade_medium(response):
    response = response.lower()
    score = 0.0

    if "15 march" in response:
        score += 0.4
    if "5 pm" in response:
        score += 0.3
    if "john" in response:
        score += 0.3

    return score


def grade_hard(response):
    response = response.lower()
    score = 0.0

    if "meeting" in response:
        score += 0.3
    if "tomorrow" in response:
        score += 0.3
    if "5 pm" in response:
        score += 0.2
    if "sure" in response or "confirm" in response:
        score += 0.2

    return score