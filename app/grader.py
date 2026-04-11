def grade(task_text, response):
    response = response.lower()

    if "classify" in task_text.lower():
        return 1.0 if "spam" in response else 0.0

    elif "extract" in task_text.lower():
        if "john" in response and "march" in response and "5" in response:
            return 1.0
        return 0.0

    elif "write" in task_text.lower():
        return 1.0 if "meeting" in response else 0.5

<<<<<<< HEAD
    return 0.0
=======
    return 0.0
>>>>>>> a9bb911 (final phase 2 fix)
