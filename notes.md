## Experiment Notes

### Exp 0 - 0417

Bugged training and validation code:
 - training code interleaved rollouts from different prompts (drive towards empirical mean)
 - validation code did not temperature decode so variance is high


### Exp 1 - 0420

Fixed training and validation code and added supervised loss
This run mainly tried to test the differences between GRPO, DAPO and DR. GRPO

Results:
 - GRPO performed best, let us stick with just GRPO moving forwards and run ablations at end

BUG!
 - supervised loss not written in the correct place

 Need to put the relevant code not in dp\_actor.py but rather workers/utils/losses.py


### Exp 2 - 0423

Fixed supervised loss bug. This run tests three things on top of the naive run from Exp 1
 - What happens if we make the prompt more detailed (as in [page 18](https://arxiv.org/pdf/2602.13540))
 - What happens if we introduce a loss that decreases probability of incorrect probability estimate R (supervised loss = (R-P)^2 * log P(c\_R) )
 - What happens if we introduce a loss that increases probability of correct probability estimate (in a gaussian area range around emperical mean P (supervised loss = - \sum a\_i * log P(c\_i)) where a\_i is a probability distribution over P +- 5 and is highest at c\_i = p)

.sum() => .mean() for aggregation (note from 0427, this was in fact, not the issue, see notes for Exp3)

### Exp 3 - 0424

Fixed loss logging and scaling issues, added a negative epsilon = 0.05 penalty to prevent entering R=0 absorbing state.
New set of experiments with the normal prompt:
    - wager\_baseline: just usual training without any supervised loss
    - calib: with soft gaussian signal towards emperical mean (k=5)
    - suppr: hard signal away from predicted values
    - comb: both calib and suppr losses

Issues with this run are that we are not correctly scaling the ratios, solution is to divide out the supervised and calib
losses by another factor of #sequences in a minibatch so that the aggregation of gradients is of the correct scale

### Exp 4 - 0427

New set of experiments with detailed prompt:
all as above
    - wager\_baseline:
    - calib: 
    - suppr:
    - comb: 

conclusion that detailed prompt does help diversity (see distribution of verbalized prompts)
suppr loss does help drive away from correct places
calib loss still attracts towards R = 0 attraction (cannot break out from pg loss..)

loss is still decreasing, maybe train for longer?? as long as we are not saturating val accuracy...
50 -> 100 steps?

### Exp 5 - 0429

Same experiments with now clipped suppression loss to get rid of attracting states.

Results:
    - attracting states are now at p=0.1 and 0.9
    - empirical capabilities are very bimodal at 0 and 100
    - baseline wager\_grpo is slowly modeling this behavior by step 100

### Exp6 - 0501

Extend training to 150 steps

### Exp 7 - 0511

Issues with training from command line updates. We are now tapering calibration at extreme ends of accuracy (0 and 100 accuracy batches)
after 30 steps. See distribution when models are trained tomorrow

Results:
    - Filtering out the probability mass at the extreme ends of accuracy resulted in all probability mass gathered at intermediate values
    - Learning rate higher is better, model needs high learning rate (can continue to experiment)
    - When the calibration lr is tapered off, this effect is convergence to distribution?
