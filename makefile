all:
	g++ main.cpp TTC_FOE.cpp -o ttc_foe `pkg-config opencv --cflags --libs`
	./ttc_foe