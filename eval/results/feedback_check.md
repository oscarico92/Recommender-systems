# Retraining with app feedback: check

Club: Manchester United. Serving model (cutoff 2026-07-01), same seed with and without feedback. Shortlisted players are added as positive edges (weight 0.5), dismissed players are hard negatives. In the API, a dismissed player is also hidden at once, before any retraining.

| Player | Action | Rank before | Rank after |
| --- | --- | --- | --- |
| Arnaut Danjuma | shortlisted | 30 | 1 |
| David Neres | shortlisted | 40 | 2 |
| Luuk de Jong | shortlisted | 50 | 3 |
| André Silva | dismissed | 1 | 5627 |
| Renato Sanches | dismissed | 2 | 5253 |

Top 10 overlap with the model without feedback, averaged over the 322 clubs: 29%. Training without feedback with another seed gives 27%: the top 10 of a club already change that much from one training run to the next, so the feedback of one club does not disturb the others more than a new training run does.
