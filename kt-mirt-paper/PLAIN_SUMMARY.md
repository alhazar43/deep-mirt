# What this paper says, in plain language

Educational software watches students answer practice questions and
uses a model to guess what each student knows. Researchers increasingly
treat the model's internal numbers as measurements. They publish them
as learning curves, as mastery scores for each skill, and as maps of
which skills help or hurt each other. The problem is how those numbers
get checked. The field checks whether the model predicts the next
answer well. Predicting answers and measuring knowledge are different
jobs. A model can predict well while its internal numbers are wrong,
and nobody would notice.

This paper builds a testing procedure that checks the numbers directly.
The idea is simple. Build a fake dataset that closely imitates a real
one, but where we secretly control the truth. We decide whether
learning happens and which skills affect which. Then we ask whether the
model, reading only the fake data, recovers what we planted. If it
cannot find the truth when we know the truth, it cannot be trusted to
find it when we do not. All pass and fail lines were written down and
locked before any test ran, so we could not move the goalposts
afterward. When a test fails, the failure is published with its cause,
not buried.

The procedure was run in full on one question and partway on a second,
and the paper says plainly which is which.

The finished question is whether a group of students learned at all.
The detector passed every planted test, on data-rich and data-poor
imitations alike. It failed once, raising a false alarm on students who
had already nearly mastered the material. We traced the cause, fixed
it, and re-tested. The fixed detector was then pointed at two real
datasets. It found learning in a large algebra dataset, on the skills
students had not already mastered, from a single run. It stayed silent
on a dataset where students attempt each skill only a handful of
times. The difference may come down to how much practice data each
student leaves behind, but the two datasets differ in other ways too,
so the paper calls this a consistent pattern rather than a proven
explanation. The run that would settle it is still going.

The second question is whether practicing one skill helps or hurts
another. Here the paper stopped partway on purpose. On fake data with
three skills, the model recovered the helping and hurting effects we
planted, and we measured the smallest effect it can reliably see. That
smallest effect is twice the size of a typical real one, and the
measurement assumes cleaner inputs than our own setup achieves. That
is all we claim. We do not claim this works on real data. We do not
claim practicing one skill causes changes in another. One of our own
checks killed the story that the model is tracking effects unfolding
over time. The checks that would earn the stronger claims are designed
and written down, but not yet run.

Just as important is what the paper refuses to say, because refusals
here are results, each with its reason attached. The procedure refuses
to endorse learning curves for individual skills, at either data
density. It refuses claims about how much students learned or which
skills grew fastest. It refuses to trust the standard model design the
field actually deploys, which failed our honesty checks on the fake
data at realistic scale. It refuses the skill-to-skill reading
entirely when questions cover several skills at once, because the
recovered directions collapse there.

Everything ends in one table. Each kind of claim gets one of four
labels. Passed on controlled data. Confirmed on real data. Refused,
with the reason. Or awaiting a test that is already designed. The
paper does not make any model predict better and never says it does.
It says which sentences about students these models have earned the
right to support, and exactly where the earned sentences stop.
