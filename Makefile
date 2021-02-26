HIPCC=/opt/rocm/hip/bin/hipcc

CFLAGS := -std=c++11 -I/opt/rocm/include/ -I./

PROGRAMS :=  bufferMatch_test 

all: $(PROGRAMS)

# Step

bufferMatch_test:  bufferMatch_test.o
	$(HIPCC) -o $@ $< 

bufferMatch_test.o: bufferMatch_test.cpp bufferMatch.hpp
	$(HIPCC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)

