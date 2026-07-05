# Q-MIRT Transfer Paper: Plain State

## What we are building

Tutoring apps show two claims to students: "you are improving" and "practicing one topic helps another." These claims are displayed as numbers, but nobody verifies that those numbers mean what the apps say. Our first paper, submitted to Computers and Education: Artificial Intelligence, showed that a standard question-response model can report numbers that are badly wrong while passing all routine error checks. This paper takes the student-side tracker and builds the simplest version whose two claims can actually be verified.

## How the model works

Every answer scores 0 to 4 points. We measure question difficulty and sensitivity once on held-out data using marginal maximum likelihood, then lock these values—the measurement ruler does not move. Each student gets one skill level per topic. A level changes only when the student practices a question: practicing a topic raises that student's skill on it (with the increase slowing as they hit their personal ceiling), and practicing topic A can shift skill on topic B by a learned amount we call the transfer coefficient. Answers do not directly push skill levels up or down. Instead, we use answers afterward to infer where each student started and how strong the mechanisms are.

## The two verifications

We verify learning with twin synthetic worlds. In the twin where skill never changes, the mechanism-based model must stay flat. In the twin where learning does happen, it must rise, and its predicted score gains must match the observed gains on the locked questions. The transfer verification trains the model, then cuts the A-to-B connection and predicts the future. If predictions on topic B get worse, the connection carried real information. In the no-learning world, cutting the connection changes nothing. Both verifications passed in every random trial at every scale we tried.

## What the work revealed

At 2000 students and 360 questions the results held. Three exploitation modes emerged: skill drift toward a fitted target, fitted decay on data where skill only rises, and gain functions too rigid to express realistic slowdown. We banned all three. Every route skill takes must trace back to a practice event. We also learned that estimating question parameters from students who are learning—with free per-student parameters—is inconsistent; measuring questions on a separate non-learning group recovers them nearly as well as oracle data. On a trajectory comparison: a standard neural tracker (LSTM) draws false progress from random variation in the no-learning world, but our mechanism model stays flat. Where learning exists, it follows more closely. The question-drift check works but needs refinement before we trust it on real data. Individual student forecasts run into a ceiling—when skill wobbles randomly from step to step, only 6–9% of the differences between students' total growth is predictable by any model—but group claims are sound.

## Mistakes we made

A sign error inverted every loss gap until code review caught it; we retracted and re-ran those claims. We first blamed the calibration failure on the incidental-parameters problem, but unequal optimization budgets in the comparison muddied the picture. The resolved finding is budget, then marginalization, then calibration-sample spread. Some numbers were reported from memory and later reconciled with saved data.

## Where we go next

We are returning to a knowledge-tracing encoder paired with item-response-theory decoding. The new design will model learning explicitly—growth will emerge from question features, concept structure, response patterns, and timing—rather than just permit and detect it. We will explore letting one question draw on several concepts at once (more realistic pedagogically) and generative training (evidence lower bound) where the transfer structure is part of the model itself.

---

Detailed numbers and the retraction ledger live in `docs/qmirt_experiment_results.md`; the prior plan of record is `docs/qmirt_paper_plan.md`.
