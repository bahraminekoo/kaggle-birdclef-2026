# Experiment Results

Track all experiments, scores, and learnings here.

## Baseline Results

### Experiment 1: EfficientNet-B0 Baseline

**Date:** [Date]  
**Configuration:**
- Model: EfficientNet-B0
- Epochs: 10
- Batch size: 32
- Learning rate: 1e-3
- Folds: 4 (0, 1, 2, 3)
- Data: train_audio only
- Augmentation: Basic (Gaussian noise, time stretch, pitch shift, time shift)

**Results:**
- Local CV: [Score]
- Public LB: 0.80
- Rank: 1037/1480

**Notes:**
- Baseline implementation
- Not using train_soundscapes data
- Simple augmentation only

---

## Phase 1 Improvements

### Experiment 2: [Name]

**Date:** [Date]  
**Configuration:**
- [Details]

**Results:**
- Local CV: [Score]
- Public LB: [Score]
- Rank: [Rank]

**Notes:**
- [Observations]

---

## Experiment Tracking Template

```markdown
### Experiment X: [Name]

**Date:** [Date]
**Configuration:**
- Model: 
- Epochs: 
- Batch size: 
- Learning rate: 
- Scheduler: 
- Folds: 
- Data sources: 
- Augmentation: 
- Special techniques: 

**Results:**
- Local CV: [Score] ± [Std]
- Public LB: [Score]
- Private LB: [Score] (after competition)
- Rank: [Rank]

**Fold Scores:**
- Fold 0: [Score]
- Fold 1: [Score]
- Fold 2: [Score]
- Fold 3: [Score]

**What Worked:**
- [Point 1]
- [Point 2]

**What Didn't Work:**
- [Point 1]
- [Point 2]

**Next Steps:**
- [Action 1]
- [Action 2]

**Notes:**
- [Additional observations]
```

---

## Leaderboard Progress

| Date | Experiment | Local CV | Public LB | Rank | Notes |
|------|------------|----------|-----------|------|-------|
| [Date] | Baseline | - | 0.80 | 1037/1480 | Initial submission |
| | | | | | |

---

## Best Submissions

1. **[Experiment Name]** - Public LB: [Score], Private LB: [Score]
2. **[Experiment Name]** - Public LB: [Score], Private LB: [Score]

---

## Lessons Learned

### What Works
- [Technique 1]
- [Technique 2]

### What Doesn't Work
- [Technique 1]
- [Technique 2]

### Key Insights
- [Insight 1]
- [Insight 2]
