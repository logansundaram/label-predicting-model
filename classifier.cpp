#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "csvstream.hpp"
using namespace std;

class Classifier{
  public://look to implement moving the print statements out and 
    Classifier(vector<string>& labels, vector<string>& contents, bool train_only)
    : labels(labels), contents(contents){
      numposts = labels.size();
      if (train_only){
        print_trainingdata();
      }
      

      string str_tot;
      for (int i = 0; i < numposts; ++i){
        str_tot += " " + contents[i] + " ";
      }

      unique_words = extract_unique_words(str_tot);
      
      cout << "trained on " << numposts << " examples" << endl; //This is for both

      if (train_only){
        print_traindatasize();
      }

      cout << "\n"; 
      //You need to fix the spacing issue





      set_posts_per_word();
      set_posts_per_label();
      
      if (train_only){
        print_trainonly();
      }

    }

    void print_trainingdata(){
      cout << "training data:" << endl;

      for (int i = 0; i < numposts; ++i){
        cout << "  label = " << labels[i] << ", content = " << contents[i] << endl;
      }
    }

    void print_traindatasize(){
      cout << "vocabulary size = " << unique_words.size() << endl;
    }

    void print_trainonly(){
      cout << "classes:" << endl;
      for (const auto& pair : posts_per_label){
        cout << "  " << pair.first << ", " << pair.second
        << " examples, log-prior = " << calc_log_prior(pair.first) << endl;
      }
      cout << "classifier parameters:"<< endl;
      for (const auto& pair : posts_per_word_label){
        if (pair.second > 0){
          cout << "  " <<pair.first.first << ":" << pair.first.second
           << ", count = " << pair.second << ", log-likelihood = " 
           << calc_log_likelihood(pair.first.first, pair.first.second) << endl;
        }
      }

      cout << "\n";
    }

    set<string> extract_unique_words(const string &str) {
      istringstream source(str);
      set<string> words;
      string word;
      while (source >> word) {
        words.insert(word);
      }
      return words;
    }

    void set_posts_per_word() {
      /*for (const string& word : unique_words) {
          for (int i = 0; i < numposts; ++i) {
              istringstream iss(contents[i]);
              string token;
              bool found = false;
  
              while (iss >> token) {
                if (token == word) {
                    found = true;  
                    break;         
                }
              }

              if (found) {
                  posts_per_word[word]++;  
                  std::pair<std::string, std::string> p = {labels[i], word};
                  posts_per_word_label[p]++; 
              }

          }
      }*/
      for (int i = 0; i < numposts; ++i){
        set<string> unqiue_words_post = extract_unique_words(contents[i]);
        for (auto& word : unqiue_words_post){
          posts_per_word[word]++;
          std::pair<std::string, std::string> p = {labels[i], word};
          posts_per_word_label[p]++; 
        }
      }

    }
  

    void set_posts_per_label(){
        for (int i = 0; i < numposts; ++i){
          posts_per_label[labels[i]]++;
        }
    }

    double calc_log_prior(const string& label){
      return log(posts_per_label[label]/(numposts + 0.0));
    }

    double calc_log_likelihood(const string& label, const string& word){
      pair<string, string> p = {label, word};
      if (posts_per_word_label[p] != 0){
        return log(posts_per_word_label[p]/(posts_per_label[label]+0.0));
      }
      else if (unique_words.find(word) != unique_words.end()){
        return log(posts_per_word[word]/(numposts+0.0));
      }
      else {
        return log(1/(numposts+0.0));
      }
      
    }

    double calc_log_probability(const string& label, const string& content){
      set<string> content_unique_words = extract_unique_words(content);
      double value = calc_log_prior(label);
      for (string word : content_unique_words){
        value += calc_log_likelihood(label, word);
      }
      return value;
    }

    pair<string, double> predict(const string& content){
      pair<string, double> value = {"", 0};
      for (const auto& pair : posts_per_label){
        double accumulation = 0;
        accumulation = calc_log_probability(pair.first, content);
        if (value.first == ""){
          value.first = pair.first;
          value.second = accumulation;
        }
        else if(value.second < accumulation){
          value.first = pair.first;
          value.second = accumulation;
        }
      }
      return value;
    }


    void test_model(vector<string>& test_labels, vector<string>& test_content){
      int numcorrect = 0;
      cout << "test data:" << endl;
      for (int i = 0; i < test_labels.size(); ++i){
        pair<string, double> values = predict(test_content[i]);
        cout << "  correct = " << test_labels[i] << ", predicted = " 
        << values.first << ", log-probability score = " << values.second << endl;
        cout << "  content = " << test_content[i] << endl;
        cout << "\n";
        if (test_labels[i] == values.first){
          numcorrect++;
        }
      }
      cout << "performance: " << numcorrect 
      << " / " << test_labels.size() << " posts predicted correctly" << endl;
    }





  private:
    vector<string> labels;
    vector<string> contents;
    int numposts;
    set<string> unique_words;
    map<string, int> posts_per_word;
    map<string, int> posts_per_label;
    map<pair<string, string>, int> posts_per_word_label;

};

int main(int argc, char** argv) {
  cout.precision(3);

  if (argc != 2 && argc != 3){
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }
  
    map<string, string> data; 
    vector<string> v_labels;
    vector<string> v_content;
    string trainfile = argv[1];
    try {
      csvstream csvin(trainfile);
  
      while (csvin >> data) {
        v_labels.push_back(data["tag"]);
        v_content.push_back(data["content"]);//dont hardcode, just for show
      }
    } catch(const csvstream_exception &e) {
      cout << "Error opening file: " << trainfile << endl;
      return 2;
    }

    if(argc == 2){
      Classifier C(v_labels, v_content, true);
    }

    if (argc == 3){//train and test
    vector<string> v_labels_test;
    vector<string> v_content_test;
    string testfile = argv[2];
    try {
      csvstream csvin_test(testfile);
  
      while (csvin_test >> data) {
        v_labels_test.push_back(data["tag"]);
        v_content_test.push_back(data["content"]);//dont hardcode, just for show
      }
    } catch(const csvstream_exception &e) {
      cout << "Error opening file: " << testfile << endl;
      return 3;
    }
    Classifier C(v_labels, v_content, false);
    C.test_model(v_labels_test, v_content_test);
  }

  
  return 0;
}