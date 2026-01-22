# MIRT Theory Notes for Deep-GPCM

## Context
Deep-GPCM currently implements a DKVMN-driven GPCM with scalar ability (theta), scalar discrimination (alpha), and ordered thresholds (beta). The extension targets multi-dimensional latent traits (MIRT) while retaining polytomous GPCM outputs.

## Notation
- D: number of latent trait dimensions
- K: number of ordinal categories
- theta_t \in R^D: student latent trait vector at time t
- a_j \in R^D: item discrimination vector for item j
- b_{j,k}: item thresholds for category k (k = 1..K-1)

## MIRT-GPCM Logits
For item j at time t:

z_{t,k} = \sum_{h=1}^{k} (a_j^T theta_t - b_{j,h})

The category probabilities are:

P(Y = k) = exp(z_{t,k}) / \sum_{c=0}^{K-1} exp(z_{t,c})

where z_{t,0} = 0 by convention.

## Trait Dynamics via DKVMN
- DKVMN read content represents a temporally evolving latent state.
- Summary network maps (read_content + q_embed) to theta_t \in R^D.
- The model predicts psychometric state rather than binary mastery.

## KC-to-Trait Mapping
Let c_j \in R^{C} be a KC vector for item j (multi-hot or weighted). Then:

- Linear map: a_j = W_kc_to_trait * c_j
- Optional bias: a_j = W * c_j + b

Alternative: learnable item embedding e_j and map to traits, with optional KC regularization.

## Constraints and Regularization
- Non-negativity (optional): enforce a_j >= 0 for interpretability.
- Norm constraints: ||a_j||_2 <= tau to reduce overfitting.
- Ordered thresholds: b_{j,1} < b_{j,2} < ... < b_{j,K-1}

## Compatibility with Existing Losses
- Losses in Deep-GPCM expect logits or probabilities of shape (batch, seq, K).
- MIRT-GPCM preserves this interface, only changing the logits computation.
