from rate_limit import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_requests_under_limit_allowed():
    limiter = RateLimiter(max_requests=20, clock=FakeClock())
    assert all(limiter.allow("u") for _ in range(20))


def test_window_expiry_resets_limit():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=2, window=60, clock=clock)
    for _ in range(5):
        limiter.allow("u")
    clock.now = 61
    assert limiter.allow("u")


def test_users_are_independent():
    limiter = RateLimiter(max_requests=1, clock=FakeClock())
    limiter.allow("a")
    limiter.allow("a")
    assert limiter.allow("b")
