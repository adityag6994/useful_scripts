//make it paralell
//how can you improve class image, what problems do you see ?
#include <vector>
#include <stdlib.h>

#include <random>

// So that I can use cout for de-bugging purposes
#include <iostream>
#include <thread>
#include <bits/stdc++.h>

using namespace std;

class Image
{
public:

  Image()
  {
    
  }

  Image(int w, int h)
  {
    width = w;
    height = h;
    image = (float *)malloc(h * w*sizeof(float));
  }

  void fill()
  {  
    // Making some random image content, just as an example. This part works fine.
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(0, 1);
    for (int i = 0; i < height * width; i++)
    {
      // image[i] = distr(eng);
      image[i] = i+10;
    }
  }

  ~Image()
  { // Should probably be virtual
      free(image);
  }

  void sobel(Image * im_result)
  {
	    // Could be constants
	    float kx[3][3] = {{-1, 0, 1},
	                      {-2, 0, 2},
	                      {-1, 0, 1}};

	    float ky[3][3] = {{-1, -2, -1},
	                      {0, 0, 0},
	                      {1, 2, 1}};

	    // A Sobel edge detector is an image filter used to isolate steep changes in an image,
	    // It produces another image where high pixel intensity values indicate
	    // a likely edge in the original image. 

	    // There are 2 filters (or kernels) here (kx, and ky) One emphisizes edges in the x direction and
	    // one emphisizes edges in the y direction. Each one will produce a new image (edge_x and edge_y).
	    // To produce the edge_x/y images, for each output pixel in edge matrices, perform the dot product
	    // with the appropriate kernel and the 3 x 3 neighborhood in the same position in the 
	    // original image

	    // The final image would be the magnitude of the 2 images, 
	    // i.e. im_result = sqrt(edge_x*edge_x + edge_y*edge_y)

	    //to initiate mask with 0 values to compensate for border pixels
	    
	    //TODO : optimize it, to avoid going through all pixels
	    for (unsigned r = 0; r < height; r++)
	    {
	      for (unsigned c = 0; c < width; c++)
	      {
	        im_result->image[r*width + c] = 0; // TODO produce edge data
	      }
	    }
	    
	    // TODO
	    for (unsigned r = 1; r < height-1; r++)
	    {
	      for (unsigned c = 1; c < width-1; c++)
	      {
	        // im_result->image[r*width + c] = 1; // TODO produce edge data
	        float p_kx = kx[0][0]*image[(r-1)*width + c-1]+
	                     kx[0][1]*image[(r-1)*width + c]+
	                     kx[0][2]*image[(r-1)*width + c+1]+
	                     kx[1][0]*image[(r)*width + c-1]+
	                     kx[1][1]*image[(r)*width + c]+
	                     kx[1][2]*image[(r)*width + c+1]+
	                     kx[2][0]*image[(r+1)*width + c-1]+
	                     kx[2][1]*image[(r+1)*width + c]+
	                     kx[2][2]*image[(r+1)*width + c+1];
	        
	        float p_ky = ky[0][0]*image[(r-1)*width + c-1]+
	                     ky[0][1]*image[(r-1)*width + c]+
	                     ky[0][2]*image[(r-1)*width + c+1]+
	                     ky[1][0]*image[(r)*width + c-1]+
	                     ky[1][1]*image[(r)*width + c]+
	                     ky[1][2]*image[(r)*width + c+1]+
	                     ky[2][0]*image[(r+1)*width + c-1]+
	                     ky[2][1]*image[(r+1)*width + c]+
	                     ky[2][2]*image[(r+1)*width + c+1];
	        
	        im_result->image[r*width + c] = sqrt(p_kx*p_kx + p_ky*p_ky);      
	        
	      }
	      std::cout << std::endl;
	    }
	    
	    // to verify 
	    for (unsigned r = 0; r < height; r++)
	    {
	      for (unsigned c = 0; c < width; c++)
	      {
	        std::cout << im_result->image[r*width + c] << " ";
	      }
	      std::cout << std::endl;
	    }

  }

  void print_vector(std::vector<float> v){
  	cout << " v : " ;
  	for(int i=0; i<v.size(); i++){
  		cout << v[i] << " " ;
  	}
  	cout << endl;
  }

  void median(Image* im_result){
  	cout << "Median Filter Starting" << endl;
  	int s=3;

  	for (unsigned r = 0; r < height; r++){
      for (unsigned c = 0; c < width; c++){
        std::cout << im_result->image[r*width + c] << " ";
      }
      std::cout << std::endl;
    }

  	for(unsigned r=0; r<height; r++){
  		for(unsigned c=0; c<width; c++){
  			vector<float> pixels;
			if(r==0 && c==0){
				//Corner : top-left
				for(unsigned p=0; p<=s/2; p++){
					for(unsigned q=0; q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c+q)]);
					}
				}
				// print_vector(pixels);
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[s/2]+pixels[(s+1)/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[s/2];
				}
			}else if(r==0 && c==(width-1)){ 
				//Corner : top-right
				for(unsigned p=0; p<=s/2; p++){
					for(unsigned q=0; q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c-q)]);
					}
				}
				// print_vector(pixels);
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[s/2]+pixels[(s+1)/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[s/2];
				}
			}else if(r==(height-1) && c==0){ 
				//Corner : bottom-left
				for(unsigned p=0; p<=s/2; p++){
					for(unsigned q=0; q<=s/2; q++){
						pixels.push_back(image[(r-p)*width + (c+q)]);
					}
				}
				// print_vector(pixels);
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[s/2]+pixels[(s+1)/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[s/2];
				}
			}else if(r==(height-1) && c==(height-1)){ 
				//Corner : bottom-left
				for(unsigned p=0; p<=s/2; p++){
					for(unsigned q=0; q<=s/2; q++){
						pixels.push_back(image[(r-p)*width + (c-q)]);
					}
				}
				// print_vector(pixels);
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[s/2]+pixels[(s+1)/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[s/2];
				}
			}else if(r==0){
				//Corner : top-row (except corners)
				for(unsigned p=0; p<=s/2; p++){
					for(int q=(-s/2); q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c+q)]);
					}
				}
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[pixels.size()/2-1]+pixels[(pixels.size())/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[pixels.size()/2];
				}
			}else if(r==height-1){
				//Corner : bottom-row (except corners)
				for(int p=0; p<=s/2; p++){
					for(int q=(-s/2); q<=s/2; q++){
						pixels.push_back(image[(r-p)*width + (c+q)]);
					}
				}
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[pixels.size()/2-1]+pixels[(pixels.size())/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[pixels.size()/2];
				}
			}else if(c==0){
				//Corner : top-row (except corners)
				for(int p=(-s/2); p<=s/2; p++){
					for(int q=0; q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c+q)]);
					}
				}
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[pixels.size()/2-1]+pixels[(pixels.size())/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[pixels.size()/2];
				}
			}else if(c==width-1){
				//Corner : top-row (except corners)
				for(int p=(-s/2); p<=s/2; p++){
					for(int q=0; q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c-q)]);
					}
				}
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[pixels.size()/2-1]+pixels[(pixels.size())/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[pixels.size()/2];
				}
			}else{
				//remaining
				for(int p=(-s/2); p<=s/2; p++){
					for(int q=(-s/2); q<=s/2; q++){
						pixels.push_back(image[(r+p)*width + (c+q)]);
					}
				}
				sort(pixels.begin(), pixels.end());
				if(pixels.size()%2==0){
					im_result->image[r*width + c] = (pixels[pixels.size()/2-1]+pixels[(pixels.size())/2])/2;
				}else{
					im_result->image[r*width + c] = pixels[pixels.size()/2];
				}
			}  			
  		}
  	}

  	for (unsigned r = 0; r < height; r++){
      for (unsigned c = 0; c < width; c++){
        std::cout << im_result->image[r*width + c] << " ";
      }
      std::cout << std::endl;
    }


  }

  float *image; 
  int width, height;
};


int main() 
{
	  const unsigned n_images = 1;
	  const unsigned width = 7;
	  const unsigned height = 7;

	  vector<Image *> original_ims;
	  vector<Image *> edge_maps;

	  for (unsigned i = 0; i < n_images; i++)
	  {
	    original_ims.push_back(new Image(width, height));
	    edge_maps.push_back(new Image(width, height));

	    original_ims[i]->fill();
	  }

	  for (unsigned i = 0; i < n_images; i++){
	    original_ims[i]->median(edge_maps[i]);
	    cout << " --- " << endl; 
	    for(unsigned r=0; r<height; r++){
	  		for(unsigned c=0; c<width; c++){
	  			std::cout << original_ims[i]->image[r*width + c] << " ";
	  		}
	  		cout << endl;
	  	}
	  }
  

}
