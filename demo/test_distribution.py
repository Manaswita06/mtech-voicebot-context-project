from demo.dataset.scenario_distribution import ScenarioDistribution

dist = ScenarioDistribution()

for i in range(10):

    primary = dist.sample_primary_intent()

    secondary = dist.sample_secondary_intent(primary)

    ambiguity = dist.sample_ambiguity()

    failure = dist.sample_tool_failure()

    sentiment = dist.sample_sentiment()

    followup = dist.sample_followup_required()

    print("=" * 60)

    print("Primary:", primary)

    print("Secondary:", secondary)

    print("Ambiguity:", ambiguity)

    print("Failure:", failure)

    print("Sentiment:", sentiment)

    print("Follow-up:", followup)