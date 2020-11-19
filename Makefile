HIPCC=/opt/rocm/hip/bin/hipcc

CFLAGS := -std=c++11

PROGRAMS :=  bufferMatch_test 

all: $(PROGRAMS)

# Step

bufferMatch_test:  bufferMatch_test.o
	$(HIPCC) -o $@ $< 

%.o: %.cpp
	$(HIPCC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)

