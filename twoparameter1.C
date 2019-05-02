#include "genann.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char *argv[])
{
  std::cout << "Train an F(c,T) function using backpropagation.\n";

  /* This will make the neural network initialize differently each run. */
  /* If you don't get a good result, try again for a different result. */
  srand(time(0));

  /* Input and expected out data for the XOR function. */
  std::vector<std::vector<double>> input;
  std::vector<double> output;
  std::ifstream in("data.dat");
  while (in)
  {
    double c1, c2, c3;
    in >> c1 >> c2 >> c3;
    input.push_back({c1, c2});
    output.push_back(c3);
  }
  std::cout << " Data loaded.\n";

  /* result buffer */
  std::vector<double> predict(output.size());

  /* shuffled indices */
  std::vector<unsigned int> index(input.size());
  for (unsigned int j = 0; j < input.size(); ++j)
    index[j] = j;

  /* New network with 2 inputs,
   * 1 hidden layer of 2 neurons,
   * and 1 output. */
  genann *ann = genann_init(2, 1, 25, 1);
  ann->activation_output = genann_act_linear;

  /* Open output files */
  std::ofstream out_history("history.dat");
  std::ofstream out_weights("weights.dat");

  /* Train on the four labeled data points many times. */
  const unsigned int nsteps = 10000000;
  double output_threshold = 1.0;
  for (unsigned int i = 0; i < nsteps; ++i)
  {
    std::random_shuffle(index.begin(), index.end());
    for (unsigned int j = 0; j < input.size(); ++j)
      genann_train(ann, &input[index[j]][0], &output[index[j]], 0.1);

    if ((i + 1) % 100 == 0 || i == nsteps - 1)
    {
      /* Run the network and see what it predicts. */
      double error = 0.0;
      for (unsigned int j = 0; j < input.size(); ++j)
      {
        predict[j] = *genann_run(ann, &input[j][0]);
        double delta = predict[j] - output[j];
        error += delta * delta;
      }

      std::cout << i + 1 << " steps. Error = " << error << '\n';
      out_history << i + 1 << ' ' << error << std::endl;

      if (error < output_threshold)
      {
        output_threshold = error * 0.5;
        std::ostringstream out_name;
        out_name << "result_" << error << ".dat";

        std::ofstream out_result(out_name.str());
        for (unsigned int j = 0; j < input.size(); ++j)
          out_result << input[j][0] << ' ' << input[j][1] << ' ' << predict[j] << '\n';
      }
    }
  }

  double error = 0.0;
  for (unsigned int j = 0; j < input.size(); ++j)
  {
    predict[j] = *genann_run(ann, &input[j][0]);
    double delta = predict[j] - output[j];
    error += delta * delta;
  }

  genann_free(ann);
  return 0;
}
