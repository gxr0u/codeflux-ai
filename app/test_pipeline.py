from app.agent.controller import handle_query


def run_test(query, code=None):
    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    response = handle_query(
        query=query,
        code=code
    )

    print("\nRESPONSE:\n")
    print(response)


if __name__ == "__main__":

    # 🔥 Test 1: Debugging (MOST IMPORTANT)
    run_test(
        query="why is my async code slow in python",
        code="""
import time

async def foo():
    time.sleep(1)
"""
    )

    # 🔥 Test 2: Concept
    run_test(
        query="what is a promise in javascript"
    )

    # 🔥 Test 3: Implementation
    run_test(
        query="how to run multiple async functions in python"
    )