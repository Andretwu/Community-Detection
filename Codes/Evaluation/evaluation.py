import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def write_File(path1,path2,num):
	path=path1
	file=[]

	with open(path) as f:		
		for line in f.readlines():
			file.append(line.strip().split(' '))	

	result=[]
	for line in file:
		for item in line:
			
			if any(char.isdigit() for char in item): #and not (any(char.isalpha() for char in item))
				result.append(item)
	
	f=open(path2,'a')
	w=''
	for i in range(len(result)):
		
		w=w+str(result[i])+','
		if (i+1)%num==0:

			f.write(w[:-1]+'\n')
			w=''
	f.close()


def read_FileIn(path):

	df=pd.read_csv(path,header=None) 
	result=[]
	
	ori=[]
	for item in df.loc[0][1:]:
		ori.append(item)
		
	result.append(ori)

	df=df.loc[1:]
	for i in range(5):
			
		data=df.loc[1+(10)*i:10+(10)*i]
		result.append((data).mean().tolist())
	
	final=[]
	for i in range(len(result)):
		item = []
		for j in range(len(result[i])):
			item.append(result[i][j])
		final.append(item)
	
	final=np.array(final).T
	
	return final


def show_PicIn(final,pic_in='InternalMetrics.png'):
	#Q0,IED(M)1,IED(SD),AE(M),AE(SD),Conductance(M)5,Conductance(SD),(RatioCut)CR(M)7,CR(SD),NC(M),NC(SD)
	
	x_1 =[0,1,2]
	x=['Modularity','Internal Edge Density','Inverse Conductance']
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080']
	labels=['GN','Louvain','USC','NSC','GA-Net']
	
	y=[]
	y.append(final[0])
	y.append(final[1])
	y.append(1-final[5])

	y=np.array(y).T
	print(y)


	plt.figure(figsize=(10,8)) 
	plt.xticks(np.arange(len(x))+0.25,x)

	for i in range(len(y[1:])): # 3 bars every time
		plt.bar([ind+0.11*i for ind in x_1], y[i+1], width=0.1,align="center",label=labels[i], color=colors[i])

	plt.bar([ind+0.11*(len(y)-1) for ind in x_1], y[0], width=0.1,align="center",label='Original', color='#000000')

	plt.legend(loc="upper right",shadow=True, prop={'size':9},labelspacing=0.2)
	plt.xlabel("Internal Metrics") 
	plt.ylabel("Value of measures")  
	plt.title("Bar Chart") 
	plt.ylim(0,1.2)
	
	plt.savefig(pic_in, dpi = 100, format = "PNG") 
	plt.show()

	return y  


def read_FileEx(path):

	df=pd.read_csv(path,header=None) 
	result=[]
	
	for i in range(5):
			
		data=df.loc[0+(10)*i:9+(10)*i]
		result.append((data).mean().tolist())
		
	final=[]
	for i in range(len(result)):
		item = []
		for j in range(len(result[i])):
			item.append(result[i][j])
		final.append(item)
	
	final=np.array(final).T
	
	return final


def show_PicEx(final,pic_ex='ExternalMetrics.png'):
	x_1=[0,1,2]
	x=['NMI','ARI','FCC']
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080']
	labels=['GN','Louvain','USC','NSC','GA-Net']

	y=final.T # every 3 metrics, total 5 types
	print(y)
	plt.figure(figsize=(10,8)) 
	plt.xticks(np.arange(len(x))+0.25,x)

	for i in range(len(y)): # 3 bars every time
		plt.bar([ind+0.11*i for ind in x_1], y[i], width=0.1,align="center",label=labels[i], color=colors[i])

	plt.legend(loc="upper right",shadow=True, prop={'size':9},labelspacing=0.2) # detalis of bars
	plt.xlabel("External Metrics") 
	plt.ylabel("Value of measures")  
	plt.title("Bar Chart") 
	
	plt.ylim(0,1.2)
	
	plt.savefig(pic_ex, dpi = 100, format = "PNG") 
	plt.show() 

	return y


def write_All(ex_y,path='result/ex_all.csv'):
	f=open(path,'a')
	for i,item in enumerate(ex_y):	
		for j,ind in enumerate(item):
			if i ==len(ex_y)-1 and j==len(item)-1:
				f.write(str(ind)+'\n')
			else:
				f.write(str(ind)+',')


def evaluation_main(resultin_path,resultex_path,evaluationin_path,evaluationex_path,pic_in,pic_ex):
	write_File(resultex_path,evaluationex_path,4)
	write_File(resultin_path,evaluationin_path,12)

	pic_ex='result/ExternalMetrics.png'
	final_ex=read_FileEx(evaluationex_path)
	ex_y=show_PicEx(final_ex,pic_ex)
	#write_All(ex_y,'result/ex_all.csv')
	
	pic_in='result/InternalMetrics.png'
	final_in=read_FileIn(evaluationin_path)
	in_y=show_PicIn(final_in,pic_in)
	#write_All(in_y,'result/in_all.csv')


if __name__ == '__main__':
	pic_ex='result/ExternalMetrics.png'
	pic_in='result/InternalMetrics.png'
	evaluation_main('result/result_in.txt','result/result_ex.txt','result/evaluation_in.csv','result/evaluation_ex.csv',pic_in,pic_ex)
