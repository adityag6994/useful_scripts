//instead of thinking in terms of corners
//think of each element as centre and expand

class Solution {
public:
    string longestPalindrome(string s) {
        // taking each element as middle check left and right side for 
        // same elements
        string result="";
        
        for(int i=0; i<s.size(); i++){
            //odd
            int left  = i-1;
            int right = i+1;
            while(left>=0 && right<s.size() 
                  && s[left]==s[right]){
                left--;
                right++;
            }
            if(result.size() < ((right-1)-(left+1)+1)){
                result = s.substr(left+1, (right-1)-(left+1)+1);
            }    
            
            //even
            left  = i;
            right = i+1;
            while(left>=0 && right<s.size() 
                  && s[left]==s[right]){
                left--;
                right++;
            }
            if(result.size() < ((right-1)-(left+1)+1)){
                result = s.substr(left+1, (right-1)-(left+1)+1);
            }    
        }
        return result;
    }
};
