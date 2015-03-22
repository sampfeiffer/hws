#include <algorithm>
#include <limits>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include "parameters.h"
#include "get_drift.h"

#include  "stdint.h"  // <--- to prevent uint64_t errors!

#include "hadoop/Pipes.hh"
#include "hadoop/TemplateFactory.hh"
#include "hadoop/StringUtils.hh"

using namespace std;

class WordCountMapper : public HadoopPipes::Mapper {
public:
    // constructor: does nothing
    WordCountMapper( HadoopPipes::TaskContext& context ) {}

    // map function: receives a line, outputs ("sum",value) and ("sum_squared",value^2)
    // to reducer.
    void map(HadoopPipes::MapContext& context)
    {
        //--- get line of text ---
        std::string value = context.getInputValue();
        float float_value = atof(value.c_str());
        if (float_value < 10){
            context.emit("sum", value);
            std::stringstream temp;
            temp << float_value*float_value;
            context.emit("sum_squared", temp.str());
        }
    }
};

class WordCountReducer : public HadoopPipes::Reducer {
public:

    double mean;

    // constructor: does nothing
    WordCountReducer(HadoopPipes::TaskContext& context) {}

    // reduce function
    void reduce( HadoopPipes::ReduceContext& context )
    {
        float get_drift(char* tick_data_filename, int &chars_per_line);
        double count = 0;
        int num_of_data = 0;

        //--- get all tuples with the same key, and count their numbers ---
        while (context.nextValue()){
            count += HadoopUtils::toFloat(context.getInputValue());
            ++num_of_data;
        }

        // Calculate and output the mean and standard deviation.
        std::stringstream temp;
        if (context.getInputKey() == "sum"){
            context.emit("num_of_data", HadoopUtils::toString(num_of_data));
            mean = count / (100.0*num_of_data);
            temp << mean;
            context.emit("mean", temp.str());
        }
        else {
            double real_sum_squared = count / 10000.0;
            double stdev = sqrt((real_sum_squared - num_of_data*mean*mean) / (num_of_data-1));
            temp << stdev;
            context.emit("stdev", temp.str());

//            // get drift
//            char* tick_data_filename = "tick_data.dat";
//            int chars_per_line = 9;
//            temp.str("");
//            temp << get_drift(tick_data_filename, chars_per_line);
//            context.emit("drift", temp.str());
        }
    }
};

int main(int argc, char *argv[])
{
    return HadoopPipes::runTask(HadoopPipes::TemplateFactory<
                                WordCountMapper,
                                WordCountReducer >() );
}
