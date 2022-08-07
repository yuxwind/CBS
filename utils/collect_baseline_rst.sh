cd $1
echo seed0
grep 'top1 are'  *_0seed/sparsity*
grep 'top5 are'  *_0seed/sparsity*
echo seed1
grep 'top1 are'  *_1seed/sparsity*
grep 'top5 are'  *_1seed/sparsity*
echo seed2
grep 'top1 are'  *_2seed/sparsity*
grep 'top5 are'  *_2seed/sparsity*
echo seed3
grep 'top1 are'  *_3seed/sparsity*
grep 'top5 are'  *_3seed/sparsity*
echo seed4
grep 'top1 are'  *_4seed/sparsity*
grep 'top5 are'  *_4seed/sparsity*

grep 'top5 are'  *_0seed/sparsity*|awk -F ', ' '{print $3}' > 0_top5
grep 'top1 are'  *_0seed/sparsity*|awk -F ', ' '{print $3}' > 0_top1
grep 'top5 are'  *_1seed/sparsity*|awk -F ', ' '{print $3}' > 1_top5
grep 'top1 are'  *_1seed/sparsity*|awk -F ', ' '{print $3}' > 1_top1
grep 'top1 are'  *_2seed/sparsity*|awk -F ', ' '{print $3}' > 2_top1
grep 'top5 are'  *_2seed/sparsity*|awk -F ', ' '{print $3}' > 2_top5
grep 'top5 are'  *_3seed/sparsity*|awk -F ', ' '{print $3}' > 3_top5
grep 'top1 are'  *_3seed/sparsity*|awk -F ', ' '{print $3}' > 3_top1
grep 'top1 are'  *_4seed/sparsity*|awk -F ', ' '{print $3}' > 4_top1
grep 'top5 are'  *_4seed/sparsity*|awk -F ', ' '{print $3}' > 4_top5
wc -l *top1
wc -l *top5
paste 0_top1 0_top5  > all.top1_top5
echo '' >> all.top1_top5
paste 1_top1 1_top5  >> all.top1_top5
echo '' >> all.top1_top5
paste 2_top1 2_top5  >> all.top1_top5
echo '' >> all.top1_top5
paste 3_top1 3_top5  >> all.top1_top5
echo '' >> all.top1_top5
paste 4_top1 4_top5  >> all.top1_top5
echo '' >> all.top1_top5

cd ..
mv $1/all.top1_top5 $1.all.top1_top5
vim $1.all.top1_top5
