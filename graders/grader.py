def calculate_score(env):
    """
    Standard OpenEnv grader entry point.
    Returns the environment's internal grader score, which is already clamped to (0.05, 0.95).
    """
    if hasattr(env, "calculate_grader_score"):
        return env.calculate_grader_score()
    return 0.5  # Fallback safe score
