#!/bin/bash
for((allnum = 1; allnum <= 30; allnum++))
do
	echo $allnum
	echo "GN Algorithm"
	times=10
	#data='../Dataset/football.txt'
	#comm='../DataSet/football_comm.dat'
	#data='../SyntheticNetworks/synthetic'$allnum'.txt'
	#comm='../SyntheticNetworks/synthetic1_comm.dat'
	cd OriginalNetwork
	python3 original.py -data '../SyntheticNetworks/synthetic'$allnum'.txt' -comm '../SyntheticNetworks/synthetic'$allnum'_comm.dat'

	cd ../GN
	rm result/time.txt
	python3 GN.py -path "result/outputofGN1.csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	for((integer = 2; integer <= $times; integer++))
	do
		cp -rf "result/outputofGN1.csv" "result/outputofGN"$integer".csv"
		#python3 GN.py -path "result/outputofGN"$integer".csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	done

	echo "Louvain Algorithm"
	cd ../Louvain
	rm result/time.txt
	python3 Louvain.py -path "result/outputofLouvain1.csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	for((integer = 2; integer <= $times; integer++))
	do
		cp -rf "result/outputofLouvain1.csv" "result/outputofLouvain"$integer".csv"
		#python3 Louvain.py -path "result/outputofLouvain"$integer".csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	done

	echo "Spectral Clustering Algorithm (Unnormalised)"
	cd ../SC
	rm result/time.txt
	for((integer = 1; integer <= $times; integer++))
	do
		python3 USC.py -path "result/outputofUSC"$integer".csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	done

	echo "Spectral Clustering Algorithm (Normalised)"
	for((integer = 1; integer <= $times; integer++))
	do
		python3 NSC.py -path "result/outputofNSC"$integer".csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	done

	echo "Genetic Algorithm (GA-Net)"
	cd ../GA-Net
	rm result/time.txt
	for((integer = 1; integer <= $times; integer++))
	do
		python3 GANet.py -path "result/outputofGA"$integer".csv" -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	done

	cd ..
	echo "Complete Data Collection !"

	echo "Starting Evaluation"
	cd Evaluation
	rm result/result_ex.txt
	rm result/result_in.txt
	rm result/evaluation_ex.csv
	rm result/evaluation_in.csv
	python3 internal.py -times $times -data '../SyntheticNetworks/synthetic'$allnum'.txt'
	python3 external.py -times $times
	python3 evaluation.py 
	cd ..
done
