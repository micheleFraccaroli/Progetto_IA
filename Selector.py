class Selector:
	def __init__(self,f):
		self.f = f

	def select(self):
		score = open(self.f,"r")
		_u = score.readline()

		alrDict = {}
		nAcc = []
		nLoss = []
		for i in score.readlines():
			n,loss,acc,lr = i.split(" ")
			nAcc.append(n)
			nLoss.append(loss)
			alrDict.update({lr : acc})

		res = max(zip(alrDict.values(), alrDict.keys()))
		id_val = list(alrDict.keys()).index(res[1])

		return res,nAcc[id_val],nLoss[id_val]

if __name__ == '__main__':
	f_name = "Score_result.txt"
	s = Selector(f_name)
	res,nAcc_id,nLoss_id = s.select()
	print("Best accurancy: " + str(res[0]) + 
		"\nLearning_rate: " + str(res[1]))