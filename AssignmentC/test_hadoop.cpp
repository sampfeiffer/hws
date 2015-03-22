#include <algorithm>
#include <limits>
#include <string>
#include <sstream>
#include <stdlib.h>

#include  "stdint.h"  // <--- to prevent uint64_t errors!

#include "hadoop/Pipes.hh"
#include "hadoop/TemplateFactory.hh"
#include "hadoop/StringUtils.hh"

using namespace std;

class WordCountMapper : public HadoopPipes::Mapper {
public:
    // constructor: does nothing
    WordCountMapper( HadoopPipes::TaskContext& context ) {}

    // map function: receives a line, outputs (word,"1")
    // to reducer.
    void map( HadoopPipes::MapContext& context )
    {
        //--- get line of text ---
        std::string value = context.getInputValue();
        float float_value = atof(value.c_str());
        if (float_value < 10){
            context.emit( "sum", value );
            std::stringstream temp;
            temp << float_value*float_value;
            context.emit( "sum_squared", temp.str());
        }
    }
};

class WordCountReducer : public HadoopPipes::Reducer {
public:
    // constructor: does nothing
    WordCountReducer(HadoopPipes::TaskContext& context) {}

    // reduce function
    void reduce( HadoopPipes::ReduceContext& context )
    {
        float count = 0;

        //--- get all tuples with the same key, and count their numbers ---
        while ( context.nextValue() ) {
          count += HadoopUtils::toFloat( context.getInputValue() );
        }

        //--- emit (word, count) ---
        context.emit(context.getInputKey(), HadoopUtils::toString( count ));
    }
};

int main(int argc, char *argv[])
{
      return HadoopPipes::runTask(HadoopPipes::TemplateFactory<
                                  WordCountMapper,
                                  WordCountReducer >() );
}
