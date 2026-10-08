"""Answer checker used by the JEE benchmark (adapted from app/utils/evaluation.py)."""


def normalize_answer(answer: str) -> str:
    return answer.lower().replace(" ", "").replace("=", "")


def is_correct(correct_answer: str, agent_solution: str) -> bool:
    correct = normalize_answer(correct_answer)
    agent = normalize_answer(agent_solution)
    return correct in agent or any(term in agent for term in correct.split())
