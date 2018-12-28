class Selector:
	def __init__(self,f):
		self.f = f

	def select(self):
		score = open(self.f,"r")
		_u = score.readline()

		alrDict = {}
		for i in score.readlines():
			loss,acc,lr = i.split(" ")
			alrDict.update({acc : lr})

		res = max(zip(alrDict.values(), alrDict.keys()))

		return res

if __name__ == '__main__':
	f_name = "Score_result.txt"
	s = Selector(f_name)
	res = s.select()
	print("Best accurancy: " + str(res[0]) + "Learning_rate: " + str(res[1]))