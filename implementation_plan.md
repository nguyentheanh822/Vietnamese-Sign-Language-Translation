# Reconstruct VSL-GH Paper in LaTeX

The user wants a perfectly formatted LaTeX paper ready for conference submission. The previous generation had several critical flaws that made it look "weird" (kì vậy) and unprofessional:
1. **Missing Tables:** Table 1, 2, 3, and 6 were completely omitted because their content wasn't in the markdown draft, causing the table numbering to skip from 2 to 4. I will reconstruct these missing tables.
2. **Missing Figure 1:** The image of the signers was removed. I will restore the figure block.
3. **Ugly Formatting:** I used `\FloatBarrier` excessively, which breaks the flow of two-column layouts and leaves massive blank spaces on the pages. I will remove `\FloatBarrier` and use standard `[t]` (top) placements to allow LaTeX to float tables beautifully.
4. **Font:** I used `FreeSerif`. I will switch to `Times New Roman` using `fontspec` to make it look exactly like a CVPR/IEEE conference paper.

## Proposed Changes
I will create a Python script that generates the complete `main.tex` file with all the correct English text from `PAPER_DRAFT_ENGLISH.md`, plus the missing tables (Table 1: International datasets, Table 2: Vietnamese datasets, Table 3: Sentence distribution, Table 6: Dataset split). 
