

################################################################################
# Computes and stores the average and current value
################################################################################
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

################################################################################
# Computes and stores the superset of added sets
################################################################################
class SortedSuperSet(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.superset = []
		self.count = 0
		self.min   = 0
		self.max   = 0

	def update(self, aset):
		new = [i for i in aset if i not in self.superset]	
		self.superset.extend(new)
		self.superset = sorted(self.superset)
		self.min = self.superset[0]
		self.max = self.superset[-1]
