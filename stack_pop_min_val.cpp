#include <iostream>
#include <vector>

using namespace std;

class stack_ds{
    public:
        std::vector<int> lt;
        int list_size=0;
        int minimum_so_far;
        vector<pair<int, int>> min_list;
        
        void push(int a){
            lt.push_back(a);
            if(list_size==0){
                minimum_so_far = a;
                min_list.push_back({lt.size()-1, a});
            }else{
                if(minimum_so_far > a){
                    //update min_list and minimum_so_far
                    minimum_so_far = a;
                    min_list.push_back({lt.size()-1, a});
                }
            }
            list_size=lt.size();
            
            for(int i=0; i<list_size; i++){
                cout << lt[i] << " ";
            }
            cout << endl;
            for(int i=0; i<min_list.size(); i++){
                cout << " [ " << min_list[i].first << "," << min_list[i].second << " ] ";
            }
            cout << endl;
        }
        
        void pop(){
            int min, stack_val;
            if(lt.size()<1){
                std::cout << "empty list!" << std::endl;
                return;
            }
            min = min_list[min_list.size()-1].second;
            if(lt.size()-1 == min_list[min_list.size()-1].first){
                min_list.pop_back();
            }
            stack_val = lt[lt.size()-1];
            lt.pop_back();
            std::cout << "min : " << min << " | stack_val : " << stack_val << std::endl;
        }
};

int main()
{
   stack_ds list;
   list.push(4);
   list.push(9);
   list.push(2);
   list.push(7);
   list.push(1);
   list.push(3);
   list.push(0);
   list.push(8);
   
  list.pop();
  list.pop();
  list.pop();
  list.pop();
  list.pop();
  list.pop();
  list.pop();
  list.pop();
  list.pop();
   
   cout << "Hello World" << endl; 
   
   return 0;
}
