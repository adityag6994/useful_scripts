// we only need to store top-k elements in set/priority_queue
//priority_queue gives access to only 1 element
//set  gives access to all the values
class Solution {
public:
    float getDist(vector<int> p){
        return sqrt(p[0]*p[0] + p[1]*p[1]);
    }
    
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        set<pair<float,int>> list;
        for(int i=0; i<points.size(); i++){
            if(list.size()<k){
                list.insert({getDist(points[i]), i});
            }else{
                auto it = prev(list.end());
                if(it->first > getDist(points[i])){
                    list.erase(it);
                    list.insert({getDist(points[i]),i});
                }
            }
        }
        vector<vector<int>> result;
        auto it = list.begin();
        for(int i=0; i<k; i++){
            vector<int> temp;
            temp.push_back(points[it->second][0]);
            temp.push_back(points[it->second][1]);
            result.push_back(temp);
            ++it;
        }
        return result;
    }
};
