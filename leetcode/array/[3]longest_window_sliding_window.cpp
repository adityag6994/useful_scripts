//use hashmap to store values with their indexes
//have pointer reffering to left of current sub-array
//update index if current element found in array 

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> visited;
        int left=0;
        int result=0;
        for(int i=0; i<s.size(); i++){
            if(visited.find(s[i])!=visited.end() && visited[s[i]]>=left){
                left = visited[s[i]]+1;
            }
            visited[s[i]]=i;
            result = max(result, i-left+1);
        }
        return result;
    }
};
