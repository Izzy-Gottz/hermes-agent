"""An absent credential must not be reported as a billing problem.

`_mark_provider_unhealthy` had one message for every caller — "payment /
credit error" — and two of its call sites reach it when a provider has no
credential at all: OpenRouter with no `OPENROUTER_API_KEY` and an exhausted
key pool, and Nous with no `auth.json`. On a machine where neither was ever
configured, the gateway log carried

    Auxiliary: marking nous unhealthy for 60s (payment / credit error)

every sixty seconds, three lines above Nous's own "no Nous authentication
found (run: hermes auth)". Anyone reading it goes to check a bill for an
account that does not exist.

These assert on the CAUSE NAMED, not merely that something was logged: a test
that only checked "a warning appeared" would have passed throughout the bug.
"""
import time
import unittest

from agent import auxiliary_client as aux


class UnconfiguredIsNotAPaymentError(unittest.TestCase):
    def setUp(self):
        aux._aux_unhealthy_until.clear()
        aux._aux_unconfigured_reported.clear()

    tearDown = setUp

    def test_an_unconfigured_provider_does_not_say_payment(self):
        with self.assertLogs(aux.logger, level="WARNING") as caught:
            aux._mark_provider_unhealthy("nous", reason="unconfigured")
        text = "\n".join(caught.output)
        self.assertIn("NOT CONFIGURED", text)
        self.assertNotIn("payment", text.lower())
        self.assertNotIn("credit", text.lower())

    def test_a_real_payment_error_still_says_payment(self):
        """The fix must not silence the case the message was written for."""
        with self.assertLogs(aux.logger, level="WARNING") as caught:
            aux._mark_provider_unhealthy("openrouter")
        self.assertIn("payment / credit error", "\n".join(caught.output))

    def test_an_unconfigured_provider_is_not_retried_on_a_one_minute_timer(self):
        """A missing credential cannot start working because 60 s passed.

        The old call sites passed ttl=60, so the chain walked into a provider
        that could never answer once a minute, for ever."""
        before = time.time()
        aux._mark_provider_unhealthy("nous", reason="unconfigured")
        held = aux._aux_unhealthy_until[aux._normalize_chain_label("nous")] - before
        self.assertGreater(held, 60 * 5)
        self.assertAlmostEqual(held, aux._AUX_UNCONFIGURED_TTL_SECONDS, delta=5)

    def test_the_explanation_is_logged_once_not_once_per_attempt(self):
        with self.assertLogs(aux.logger, level="WARNING"):
            aux._mark_provider_unhealthy("nous", reason="unconfigured")
        with self.assertLogs(aux.logger, level="DEBUG") as second:
            aux._mark_provider_unhealthy("nous", reason="unconfigured")
        self.assertEqual([r for r in second.records if r.levelname == "WARNING"], [])

    def test_an_explicit_ttl_still_wins(self):
        """Callers that name a TTL are not overridden by the reason."""
        before = time.time()
        aux._mark_provider_unhealthy("nous", ttl=30, reason="unconfigured")
        held = aux._aux_unhealthy_until[aux._normalize_chain_label("nous")] - before
        self.assertAlmostEqual(held, 30, delta=5)


if __name__ == "__main__":
    unittest.main()
