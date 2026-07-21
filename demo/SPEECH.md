# Voice-over script — CCT Estimator demo

> Kokoro TTS input. Keep sentences short. Commas and periods control pacing.

---

This is the Critical Clearing Time estimator — a real-time stability monitoring tool for power grid operators.

Critical Clearing Time, or CCT, is the maximum duration a short-circuit fault can persist on the grid before generators lose synchronism with each other. When that happens, the grid splits. The lower the CCT, the less time operators have to react, and the higher the risk.

The estimator takes the current operating state of the grid — voltages, power flows, and network topology — and instantly returns stability estimates for all monitored scenarios.

It answers two practical questions for the operator.

First: which generators are critical right now? A generator is considered critical if a nearby fault would cause it to be the first to fall out of step. Knowing which generator is most at risk tells operators where the system is most vulnerable.

Second: which fault locations are dangerous? Each line or busbar in the network has an estimated CCT for the current state. Locations with a low CCT are flagged — because even a brief fault there could trigger a cascade.

Together, these two views give grid operators a continuously updated picture of where the system stands, so they can take preventive action before a disturbance turns into an incident.
