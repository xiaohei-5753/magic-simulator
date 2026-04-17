#pragma GCC optimize(3,"Ofast","inline")
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <queue>
#include <complex>
#include "../include/common.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;

struct pixel{
	float r,g,b,a;
	float lr,lg,lb,l;
	pixel(
		float r=0,float g=0,float b=0,float a=0,
		float lr=0,float lg=0,float lb=0
		):r(r),g(g),b(b),a(a),lr(lr),lg(lg),lb(lb){}
};

// HSL to RGB conversion (from magic project)
inline uint32_t hsl2rgb(float h, float s, float l) {
    // Normalize hue to 0-360
    h = fmod(h, 360.0f);
    if (h < 0) h += 360.0f;
    
    float c = (1.0f - fabs(2.0f * l - 1.0f)) * s;
    float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = l - c / 2.0f;
    
    float r, g, b;
    if (h < 60) {
        r = c; g = x; b = 0;
    } else if (h < 120) {
        r = x; g = c; b = 0;
    } else if (h < 180) {
        r = 0; g = c; b = x;
    } else if (h < 240) {
        r = 0; g = x; b = c;
    } else if (h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    uint8_t ru = static_cast<uint8_t>((r + m) * 255.0f);
    uint8_t gu = static_cast<uint8_t>((g + m) * 255.0f);
    uint8_t bu = static_cast<uint8_t>((b + m) * 255.0f);
    return (ru << 16) | (gu << 8) | bu;
}

// Magic color conversion constants (from magic project)
constexpr double PI_1 = 0.31830988618379067153776752674503;  // 1/π
constexpr double ARGBIAS1 = -0.11111111111111111111111111111111;  // -1/9
constexpr double ARGBIAS2 = 1;

const int W=128;
const int H=128;
const int scrW=640*2;
const int scrH=640*2;
const double eps=1e-6;
const double eps_l=1e-3;
const int R=std::max(W,H);
const int pps=4*R;

struct PrecomputedDirection {
    int dx, dy;
    float udx, udy; // 归一化方向
    int octant;     // 八分域变换信息 (3 bits: bit0: y<0, bit1: x<0, bit2: y>x)
    int absDx, absDy; // 绝对值，已排序确保absDy <= absDx
};
std::vector<PrecomputedDirection> precomputedDirs;
int totalSamples = 0;

bool flag_upscr=1;
bool flag_opscr=1;

int filecnt=0;
camera cam;
pixel wallcl={0,0,0,1,.2,.2,.2};
coordD sundr={std::sqrt(.5),std::sqrt(.5)};
float sunsc=5e-3;
pixel suncl={0,0,0,0,1*10.,1*10.,1*10.};
// pixel cvs[W][H]; // 旧数据结构
// SoA (Structure of Arrays) 优化
float cvs_r[W][H];
float cvs_g[W][H];
float cvs_b[W][H];
float cvs_a[W][H];
float cvs_lr[W][H];
float cvs_lg[W][H];
float cvs_lb[W][H];

uint8_t bk[W][H][3];

// Magic simulation variables
uint8_t cv[W_MAX][H_MAX];                 // canvas 0=empty,1=solid
Complex fi[W_MAX][H_MAX];                 // complex field
uint32_t cv_r[W_MAX][H_MAX];              // canvas with color
uint16_t cv_s[W_MAX][H_MAX];              // sprk temp
std::queue<coordI> sprk;
std::queue<Complex> sprk_f;
Complex magic_hand;
coordI magic_hand_c;
uint32_t magic_hand_r;

uint8_t clip_cv[W_MAX][H_MAX];
Complex clip_fi[W_MAX][H_MAX];
uint32_t clip_cv_r[W_MAX][H_MAX];
int clipW, clipH;
coordI clip_c;
int flag_copy;
int flag_copyx;
int flag_paste;

uint32_t color_cfg[] = {
    0xFFDEDEDE,  // 0=background
    0xFFFF0000,  // 1=real part (red)
    0xFF0000FF,  // 2=imaginary part (blue)
    0xFF000000   // 3=font color
};
double color_cfgD[sizeof(color_cfg)/sizeof(uint32_t)][4];

// Global transparency (user requirement)
float globalAlpha = 0.5f;
bool useEmission = true;

GLuint computeProgram = 0;
GLuint cvsSSBO = 0;
GLuint dirSSBO = 0;
bool flag_gpu_recompute = true;
bool DEBUG_FORCE_CPU = false; // 调试标志：强制使用CPU路径
coordI ms,prems;
string pen_s="0 0 0 1 0 0 0";

// Magic simulation functions
void magic_init();
void magic_update();
void magic_conduct();
void magic_reco_ln(int x, int y, int dir);
void magic_reco_exp(int x, int y, int dir);
void magic_parsing();
Complex magic_getCellValue(coordI cell);
void magic_setMagicHand(const Complex& value);
uint32_t magic_c2col(const Complex& c);
uint32_t magic_c2col1(const Complex& c);
void magic_mapToCvs(); // Map magic field to cvs arrays for rendering
void loadConfig();

inline bool lim_cvs(coordI p);
coordD mem2abs(coordD p);
coordD abs2mem(coordD p);
coordD abs2scr(coordD p,camera cam);
coordD scr2abs(coordD p,camera cam);
coordI mem2scr(coordI p,camera cam);
coordI scr2mem(coordI p,camera cam);
void line_cvs(coordI a,coordI b);
void circle_cvs(coordI o,int r);
pixel ray(coordI o,coordI a);
pixel ray_optimized(coordI o, const PrecomputedDirection& dir);
void sample_pix(coordI p, uint8_t& r, uint8_t& g, uint8_t& b);
pixel pen(coordD p);
float solv_expr(const string& expr,int l,int r,const coordD& p);
void precomputeDirections();

void init();
void upWI();
void upOI();
void sscr();

GLFWwindow* window = nullptr;
GLuint textureID;
GLuint vao, vbo, ebo;
GLuint shaderProgram;
bool mouse_l_down = false;
bool mouse_r_down = false;
double mouse_x = 0, mouse_y = 0;

void initGL();
void updateTexture();
void renderQuad();
void cleanupGL();
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
uint32_t EGERGB(int r,int g,int b){return (r<<16)|(g<<8)|b;}

int main(){
	if (!glfwInit()) {
		cerr << "Failed to initialize GLFW\n";
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(scrW, scrH, "2D Renderer OpenGL", NULL, NULL);
	if (!window) {
		cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW\n";
		glfwTerminate();
		return -1;
	}

	initGL();
	init();

	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);

	pen_s="0 0 0 255 0 0 0";

	while (!glfwWindowShouldClose(window)) {
		// Update magic simulation
		magic_update();
		magic_mapToCvs();
		
		// Update window input and render
		upWI();
		upOI();
		sscr();

		glfwPollEvents();
	}

	cleanupGL();
	glfwTerminate();
	return 0;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if(action==GLFW_PRESS||action==GLFW_REPEAT){
		if(mods==GLFW_MOD_CONTROL){
			if(key==GLFW_KEY_UP) {cam.sk/=1.03;flag_upscr=1;}
			if(key==GLFW_KEY_DOWN) {cam.sk*=1.03;flag_upscr=1;}
			if(key==GLFW_KEY_0) {cam=camera();flag_upscr=1;}
			if(key==GLFW_KEY_N) {init();}
		}else{
			if(key==GLFW_KEY_LEFT) {cam.x-=.02*cam.sk;flag_upscr=1;}
			if(key==GLFW_KEY_UP) {cam.y+=.02*cam.sk;flag_upscr=1;}
			if(key==GLFW_KEY_RIGHT) {cam.x+=.02*cam.sk;flag_upscr=1;}
			if(key==GLFW_KEY_DOWN) {cam.y-=.02*cam.sk;flag_upscr=1;}
			if(key==GLFW_KEY_P) {flag_upscr=1;}
			if(key==GLFW_KEY_O){cout << "input color: ";getline(cin,pen_s);}
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods){
	if(button==GLFW_MOUSE_BUTTON_LEFT){
		mouse_l_down = (action==GLFW_PRESS);
	}
	if(button==GLFW_MOUSE_BUTTON_RIGHT){
		mouse_r_down = (action==GLFW_PRESS);
	}
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
	mouse_x = xpos;
	mouse_y = ypos;
}

void upOI(){
	if (DEBUG_FORCE_CPU || computeProgram == 0) {
    std::cout << "Compute shader not available, falling back to CPU computation." << std::endl;
    // 计算着色器未初始化，回退到CPU计算
		const int BLOCK = 16;
		#pragma omp parallel for schedule(guided, 16)
		for(int by=0;by<H;by+=BLOCK){
			for(int bx=0;bx<W;bx+=BLOCK){
				int y_end = std::min(by+BLOCK, H);
				int x_end = std::min(bx+BLOCK, W);
				for(int y=by;y<y_end;y++){
					for(int x=bx;x<x_end;x++){
						sample_pix({x,y}, bk[x][y][0], bk[x][y][1], bk[x][y][2]);
	}
	flag_gpu_recompute = true;
}
			}
		}
		flag_upscr=1;
		return;
	}

  if (flag_gpu_recompute) {
    std::cout << "Using GPU compute shader for rendering." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // 上传CVS数据到SSBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, cvsSSBO);
		float* cvsData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		if (cvsData) {
			size_t offset = 0;
			// 复制cvs_r
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_r[x][y];
				}
			}
			// 复制cvs_g
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_g[x][y];
				}
			}
			// 复制cvs_b
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_b[x][y];
				}
			}
			// 复制cvs_a
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_a[x][y];
				}
			}
			// 复制cvs_lr
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_lr[x][y];
				}
			}
			// 复制cvs_lg
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_lg[x][y];
				}
			}
			// 复制cvs_lb
			for (int y = 0; y < H; ++y) {
				for (int x = 0; x < W; ++x) {
					cvsData[offset++] = cvs_lb[x][y];
				}
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// 绑定图像纹理用于写入
		glBindImageTexture(0, textureID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

		// 绑定SSBO到绑定点1
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cvsSSBO);
		// 绑定方向SSBO到绑定点2
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dirSSBO);

		// 使用计算着色器
		glUseProgram(computeProgram);
		// 设置uniform totalSamples
		glUniform1i(glGetUniformLocation(computeProgram, "totalSamples"), totalSamples);
		glDispatchCompute((W + 15) / 16, (H + 15) / 16, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "GPU compute time: " << elapsed.count() * 1000.0 << " ms" << std::endl;

		flag_gpu_recompute = false;
		flag_upscr = true;
	}
}

void init(){
	loadConfig();
	precomputeDirections();
	
	// Initialize magic simulation
	magic_init();
	
	// 上传方向数据到GPU（如果SSBO已创建）
	if (dirSSBO != 0) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirSSBO);
		size_t dirDataSize = totalSamples * 5 * sizeof(int);
		glBufferData(GL_SHADER_STORAGE_BUFFER, dirDataSize, NULL, GL_DYNAMIC_DRAW);
		int* dirData = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		if (dirData) {
			size_t idx = 0;
			for (const auto& dir : precomputedDirs) {
				dirData[idx++] = dir.dx;
				dirData[idx++] = dir.dy;
				dirData[idx++] = dir.octant;
				dirData[idx++] = dir.absDx;
				dirData[idx++] = dir.absDy;
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		std::cout << "Direction data uploaded to GPU: " << totalSamples << " samples" << std::endl;
	}
	for(int y=0;y<H;y++){
		for(int x=0;x<W;x++){
			cvs_r[x][y] = 0.0f;
			cvs_g[x][y] = 0.0f;
			cvs_b[x][y] = 0.0f;
			cvs_a[x][y] = 0.0f;
			cvs_lr[x][y] = 0.0f;
			cvs_lg[x][y] = 0.0f;
			cvs_lb[x][y] = 0.0f;
		}
	}
	pen_s="~ x 1 y 1 x 1 y";
	coordI i;
	for(i.y=0;i.y<H;i.y++){
		for(i.x=0;i.x<W;i.x++){
			if(rand()%64==0){
				pixel p = pen(mem2abs(coordD{i.x,i.y}));
				cvs_r[i.x][i.y] = p.r;
				cvs_g[i.x][i.y] = p.g;
				cvs_b[i.x][i.y] = p.b;
				cvs_a[i.x][i.y] = p.a;
				cvs_lr[i.x][i.y] = p.lr;
				cvs_lg[i.x][i.y] = p.lg;
				cvs_lb[i.x][i.y] = p.lb;
			}
		}
	}
	flag_gpu_recompute = true;
}

void precomputeDirections() {
    precomputedDirs.clear();
    int m = 1 - R;
    coordI i = {0, R};
    float invR = 1.0f / R;
    while (i.x < i.y) {
        // 8个对称方向
        int dirs[8][2] = {
            {i.x, i.y}, {i.y, i.x},
            {-i.x, i.y}, {-i.y, i.x},
            {i.x, -i.y}, {i.y, -i.x},
            {-i.x, -i.y}, {-i.y, -i.x}
        };
        for (int d = 0; d < 8; ++d) {
            PrecomputedDirection pd;
            pd.dx = dirs[d][0];
            pd.dy = dirs[d][1];
            pd.udx = pd.dx * invR;
            pd.udy = pd.dy * invR;
            
            // 计算八分域信息
            int octant = 0;
            int adx = pd.dx, ady = pd.dy;
            if (ady < 0) { octant |= 0b001; ady = -ady; }
            if (adx < 0) { octant |= 0b010; adx = -adx; }
            if (ady > adx) { octant |= 0b100; std::swap(adx, ady); }
            pd.octant = octant;
            pd.absDx = adx;
            pd.absDy = ady;
            
            precomputedDirs.push_back(pd);
        }
        i.x++;
        if (m < 0) m += 2 * i.x + 1;
        else { i.y--; m += 2 * (i.x - i.y) + 1; }
    }
    totalSamples = precomputedDirs.size();
}

void upWI(){
	ms.x = (int)mouse_x;
	ms.y = (int)mouse_y;
	coordI p=scr2mem({ms.x,ms.y},cam);
	if(!flag_opscr) {prems=ms;return;}
	if(mouse_l_down){
		line_cvs(scr2mem(prems,cam),scr2mem(ms,cam));
		flag_upscr=1;
	}
	if(mouse_r_down){
		string tmp=pen_s;
		pen_s="0 0 0 0 0 0 0";
		line_cvs(scr2mem(prems,cam),scr2mem(ms,cam));
		pen_s=tmp;
		flag_upscr=1;
	}
	prems=ms;
}

void sscr(){
	if(!flag_upscr){
		glfwWaitEventsTimeout(0.016);
		return;
	}
	updateTexture();
	renderQuad();
	flag_upscr=0;
}

void sample_pix(coordI p, uint8_t& r, uint8_t& g, uint8_t& b){
	pixel res;
	if(1 - cvs_a[p.x][p.y] < eps_l){
		res.r = cvs_r[p.x][p.y];
		res.g = cvs_g[p.x][p.y];
		res.b = cvs_b[p.x][p.y];
		res.a = cvs_a[p.x][p.y];
		res.lr = cvs_lr[p.x][p.y];
		res.lg = cvs_lg[p.x][p.y];
		res.lb = cvs_lb[p.x][p.y];
	}else{
		for(const auto& dir : precomputedDirs){
			pixel pix = ray_optimized(p, dir);
			res.r += pix.lr; res.g += pix.lg; res.b += pix.lb;
		}
		float invTotal = 1.0f / totalSamples;
		res.r *= invTotal; res.g *= invTotal; res.b *= invTotal;
		res.r = std::max(res.r,0.f); res.g = std::max(res.g,0.f); res.b = std::max(res.b,0.f);
		float over = std::max({res.r, res.g, res.b});
		if(over > 1) { res.r /= over; res.g /= over; res.b /= over; }
	}
	r = (uint8_t)(res.r * 255);
	g = (uint8_t)(res.g * 255);
	b = (uint8_t)(res.b * 255);
}

pixel ray(coordI o,coordI a){
	double l=R;coordD dr={a.x/l,a.y/l};
	int tmp=0;
	if(a.y<0) {a.y=-a.y;tmp|=0b001;}
	if(a.x<0) {a.x=-a.x;tmp|=0b010;}
	if(a.y>a.x) {swap(a.y,a.x);tmp|=0b100;}
	int m=0;int y=0;
	coordI p;
	pixel lt={0,0,0,0,1,1,1};
	for(int x=0;;x++){
		p={x,y};
		if(tmp&0b100) swap(p.x,p.y);
		if(tmp&0b010) p.x=-p.x;
		if(tmp&0b001) p.y=-p.y;
		p=p+o;
		if(p.x<0||p.x>=W||p.y<0||p.y>=H){
			double dotp=1-(sundr.x*dr.x+sundr.y*dr.y);
			if(dotp<=sunsc){
				lt.lr*=suncl.lr;lt.lg*=suncl.lg;lt.lb*=suncl.lb;
				return lt;
			}
			lt.lr*=wallcl.lr;lt.lg*=wallcl.lg;lt.lb*=wallcl.lb;
			return lt;
		}
		if(lt.lr<eps_l&&lt.lg<eps_l&&lt.lb<eps_l) return lt;
 		if(1-cvs_a[p.x][p.y]<eps_l){
			lt.lr*=cvs_lr[p.x][p.y];lt.lg*=cvs_lg[p.x][p.y];lt.lb*=cvs_lb[p.x][p.y];
			return lt;
		}
		m+=a.y;if(m>=a.x){y++;m-=a.x;}
	}
}

pixel ray_optimized(coordI o, const PrecomputedDirection& dir){
    // 使用预计算的归一化方向
    coordD dr = {dir.udx, dir.udy};
    int octant = dir.octant;
    int absDx = dir.absDx;
    int absDy = dir.absDy;
    
    int m = 0;
    int y = 0;
    coordI p;
    pixel lt = {0,0,0,0,1,1,1};
    
    for(int x = 0; ; x++){
        // 根据八分域变换计算实际坐标
        p.x = x;
        p.y = y;
        
        // 应用八分域变换
        if(octant & 0b100) std::swap(p.x, p.y);
        if(octant & 0b010) p.x = -p.x;
        if(octant & 0b001) p.y = -p.y;
        
        p = p + o;
        
        if(p.x < 0 || p.x >= W || p.y < 0 || p.y >= H){
            double dotp = 1 - (sundr.x * dr.x + sundr.y * dr.y);
            if(dotp <= sunsc){
                lt.lr *= suncl.lr; lt.lg *= suncl.lg; lt.lb *= suncl.lb;
                return lt;
            }
            lt.lr *= wallcl.lr; lt.lg *= wallcl.lg; lt.lb *= wallcl.lb;
            return lt;
        }
        if(lt.lr < eps_l && lt.lg < eps_l && lt.lb < eps_l) return lt;
        if(1 - cvs_a[p.x][p.y] < eps_l){
            lt.lr *= cvs_lr[p.x][p.y]; lt.lg *= cvs_lg[p.x][p.y]; lt.lb *= cvs_lb[p.x][p.y];
            return lt;
        }
        m += absDy;
        if(m >= absDx){ y++; m -= absDx; }
    }
}

float solv_expr(const string& expr,int l,int r,const coordD& p){
	if(l==r){
		if(expr[l]=='x') return p.x;if(expr[l]=='y') return p.y;
	}
	if(expr[l]=='('&&expr[r]==')') return solv_expr(expr,l+1,r-1,p);
	int i=l;char op=0;int op_i=-1;
	while(i<=r){
		while(i<=r?expr[i]!='+'&&expr[i]!='-'&&expr[i]!='*'&&expr[i]!='/'&&expr[i]!='^':0){
			if(expr[i]=='('){
				int cc=1;
				while(i<=r&&cc){
					i++;
					if(expr[i]=='(') cc++;
					if(expr[i]==')') cc--;
				}
				if(cc!=0) {cout << "ERROR 1";}
			}
			i++;
		}
		if(i>r) break;
		if(op=='+') {i++;continue;}
		if(expr[i]=='+') {op='+';op_i=i;i++;continue;}
		if(op=='-') {i++;continue;}
		if(expr[i]=='-') {op='-';op_i=i;i++;continue;}
		if(op=='*') {i++;continue;}
		if(expr[i]=='*') {op='*';op_i=i;i++;continue;}
		if(op=='/') {i++;continue;}
		if(expr[i]=='/') {op='/';op_i=i;i++;continue;}
		if(op=='^') {i++;continue;}
		if(expr[i]=='^') {op='^';op_i=i;i++;continue;}
	}
	if(op==0) return std::stof(expr.substr(l,r+1));
	if(op=='+') return solv_expr(expr,l,op_i-1,p)+solv_expr(expr,op_i+1,r,p);
	if(op=='-') return solv_expr(expr,l,op_i-1,p)-solv_expr(expr,op_i+1,r,p);
	if(op=='*') return solv_expr(expr,l,op_i-1,p)*solv_expr(expr,op_i+1,r,p);
	if(op=='/') return solv_expr(expr,l,op_i-1,p)/solv_expr(expr,op_i+1,r,p);
	if(op=='^') return std::powf(solv_expr(expr,l,op_i-1,p),solv_expr(expr,op_i+1,r,p));
	return 0.f;
}

inline bool lim_cvs(coordI p){
	return p.x>=0&&p.x<W&&p.y>=0&&p.y<H;
}

coordD mem2abs(coordD p) {return {p.x/W,p.y/H};}
coordD abs2mem(coordD p) {return {p.x*W,p.y*H};}
coordD abs2scr(coordD p,camera cam) {return {scrW*((p.x-cam.x)/cam.sk+.5),scrH*((cam.y-p.y)/cam.sk+.5)};}
coordD scr2abs(coordD p,camera cam) {return {cam.sk*(p.x/scrW-.5)+cam.x,cam.sk*(.5-p.y/scrH)+cam.y};}
coordI mem2scr(coordI p,camera cam){
	coordI i={scrW*((p.x/W-cam.x)/cam.sk+.5),scrH*((cam.y-p.y/H)/cam.sk+.5)};
	return {std::max(std::min(i.x,W-1),0),std::max(std::min(i.y,H-1),0)};
}
coordI scr2mem(coordI p,camera cam){
	coordI i={(cam.sk*((double)p.x/scrW-.5)+cam.x)*W,(cam.sk*(.5-(double)p.y/scrH)+cam.y)*H};
	return {std::max(std::min(i.x,W-1),0),std::max(std::min(i.y,H-1),0)};
}

pixel pen(coordD p){
	stringstream ss;
	ss << pen_s;
	float r,g,b,a,lr,lg,lb;
	if(pen_s[0]=='~'){
		string expr;ss >> expr;
		ss >> expr;r=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;g=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;b=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;a=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;lr=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;lg=solv_expr(expr,0,expr.size()-1,p);
		ss >> expr;lb=solv_expr(expr,0,expr.size()-1,p);
	}
	else{
		ss >> r;ss >> g;ss >> b;ss >> a;ss >> lr;ss >> lg;ss >> lb;
		r/=255;g/=255;b/=255;a/=255;lr/=255;lg/=255;lb/=255;
	}
	return {r,g,b,a,lr,lg,lb};
}

void circle_cvs(coordI o,int r){
	int m=1-r;
	coordI i={0,r};
	coordD p;
	while(i.x<i.y){
		// 8个对称点
		p=mem2abs({o.x+i.x,o.y+i.y});
		if(lim_cvs({o.x+i.x,o.y+i.y})) {
			pixel px = pen(p);
			cvs_r[o.x+i.x][o.y+i.y] = px.r;
			cvs_g[o.x+i.x][o.y+i.y] = px.g;
			cvs_b[o.x+i.x][o.y+i.y] = px.b;
			cvs_a[o.x+i.x][o.y+i.y] = px.a;
			cvs_lr[o.x+i.x][o.y+i.y] = px.lr;
			cvs_lg[o.x+i.x][o.y+i.y] = px.lg;
			cvs_lb[o.x+i.x][o.y+i.y] = px.lb;
			sample_pix({o.x+i.x,o.y+i.y}, bk[o.x+i.x][o.y+i.y][0], bk[o.x+i.x][o.y+i.y][1], bk[o.x+i.x][o.y+i.y][2]);
		}
		p=mem2abs({o.x+i.x,o.y-i.y});
		if(lim_cvs({o.x+i.x,o.y-i.y})) {
			pixel px = pen(p);
			cvs_r[o.x+i.x][o.y-i.y] = px.r;
			cvs_g[o.x+i.x][o.y-i.y] = px.g;
			cvs_b[o.x+i.x][o.y-i.y] = px.b;
			cvs_a[o.x+i.x][o.y-i.y] = px.a;
			cvs_lr[o.x+i.x][o.y-i.y] = px.lr;
			cvs_lg[o.x+i.x][o.y-i.y] = px.lg;
			cvs_lb[o.x+i.x][o.y-i.y] = px.lb;
			sample_pix({o.x+i.x,o.y-i.y}, bk[o.x+i.x][o.y-i.y][0], bk[o.x+i.x][o.y-i.y][1], bk[o.x+i.x][o.y-i.y][2]);
		}
		p=mem2abs({o.x-i.x,o.y+i.y});
		if(lim_cvs({o.x-i.x,o.y+i.y})) {
			pixel px = pen(p);
			cvs_r[o.x-i.x][o.y+i.y] = px.r;
			cvs_g[o.x-i.x][o.y+i.y] = px.g;
			cvs_b[o.x-i.x][o.y+i.y] = px.b;
			cvs_a[o.x-i.x][o.y+i.y] = px.a;
			cvs_lr[o.x-i.x][o.y+i.y] = px.lr;
			cvs_lg[o.x-i.x][o.y+i.y] = px.lg;
			cvs_lb[o.x-i.x][o.y+i.y] = px.lb;
			sample_pix({o.x-i.x,o.y+i.y}, bk[o.x-i.x][o.y+i.y][0], bk[o.x-i.x][o.y+i.y][1], bk[o.x-i.x][o.y+i.y][2]);
		}
		p=mem2abs({o.x-i.x,o.y-i.y});
		if(lim_cvs({o.x-i.x,o.y-i.y})) {
			pixel px = pen(p);
			cvs_r[o.x-i.x][o.y-i.y] = px.r;
			cvs_g[o.x-i.x][o.y-i.y] = px.g;
			cvs_b[o.x-i.x][o.y-i.y] = px.b;
			cvs_a[o.x-i.x][o.y-i.y] = px.a;
			cvs_lr[o.x-i.x][o.y-i.y] = px.lr;
			cvs_lg[o.x-i.x][o.y-i.y] = px.lg;
			cvs_lb[o.x-i.x][o.y-i.y] = px.lb;
			sample_pix({o.x-i.x,o.y-i.y}, bk[o.x-i.x][o.y-i.y][0], bk[o.x-i.x][o.y-i.y][1], bk[o.x-i.x][o.y-i.y][2]);
		}
		p=mem2abs({o.x+i.y,o.y+i.x});
		if(lim_cvs({o.x+i.y,o.y+i.x})) {
			pixel px = pen(p);
			cvs_r[o.x+i.y][o.y+i.x] = px.r;
			cvs_g[o.x+i.y][o.y+i.x] = px.g;
			cvs_b[o.x+i.y][o.y+i.x] = px.b;
			cvs_a[o.x+i.y][o.y+i.x] = px.a;
			cvs_lr[o.x+i.y][o.y+i.x] = px.lr;
			cvs_lg[o.x+i.y][o.y+i.x] = px.lg;
			cvs_lb[o.x+i.y][o.y+i.x] = px.lb;
			sample_pix({o.x+i.y,o.y+i.x}, bk[o.x+i.y][o.y+i.x][0], bk[o.x+i.y][o.y+i.x][1], bk[o.x+i.y][o.y+i.x][2]);
		}
		p=mem2abs({o.x+i.y,o.y-i.x});
		if(lim_cvs({o.x+i.y,o.y-i.x})) {
			pixel px = pen(p);
			cvs_r[o.x+i.y][o.y-i.x] = px.r;
			cvs_g[o.x+i.y][o.y-i.x] = px.g;
			cvs_b[o.x+i.y][o.y-i.x] = px.b;
			cvs_a[o.x+i.y][o.y-i.x] = px.a;
			cvs_lr[o.x+i.y][o.y-i.x] = px.lr;
			cvs_lg[o.x+i.y][o.y-i.x] = px.lg;
			cvs_lb[o.x+i.y][o.y-i.x] = px.lb;
			sample_pix({o.x+i.y,o.y-i.x}, bk[o.x+i.y][o.y-i.x][0], bk[o.x+i.y][o.y-i.x][1], bk[o.x+i.y][o.y-i.x][2]);
		}
		p=mem2abs({o.x-i.y,o.y+i.x});
		if(lim_cvs({o.x-i.y,o.y+i.x})) {
			pixel px = pen(p);
			cvs_r[o.x-i.y][o.y+i.x] = px.r;
			cvs_g[o.x-i.y][o.y+i.x] = px.g;
			cvs_b[o.x-i.y][o.y+i.x] = px.b;
			cvs_a[o.x-i.y][o.y+i.x] = px.a;
			cvs_lr[o.x-i.y][o.y+i.x] = px.lr;
			cvs_lg[o.x-i.y][o.y+i.x] = px.lg;
			cvs_lb[o.x-i.y][o.y+i.x] = px.lb;
			sample_pix({o.x-i.y,o.y+i.x}, bk[o.x-i.y][o.y+i.x][0], bk[o.x-i.y][o.y+i.x][1], bk[o.x-i.y][o.y+i.x][2]);
		}
		p=mem2abs({o.x-i.y,o.y-i.x});
		if(lim_cvs({o.x-i.y,o.y-i.x})) {
			pixel px = pen(p);
			cvs_r[o.x-i.y][o.y-i.x] = px.r;
			cvs_g[o.x-i.y][o.y-i.x] = px.g;
			cvs_b[o.x-i.y][o.y-i.x] = px.b;
			cvs_a[o.x-i.y][o.y-i.x] = px.a;
			cvs_lr[o.x-i.y][o.y-i.x] = px.lr;
			cvs_lg[o.x-i.y][o.y-i.x] = px.lg;
			cvs_lb[o.x-i.y][o.y-i.x] = px.lb;
			sample_pix({o.x-i.y,o.y-i.x}, bk[o.x-i.y][o.y-i.x][0], bk[o.x-i.y][o.y-i.x][1], bk[o.x-i.y][o.y-i.x][2]);
		}
		i.x++;
		if(m<0) m+=2*i.x+1;
		else {i.y--;m+=2*(i.x-i.y)+1;}
	}
	flag_gpu_recompute = true;
}

void line_cvs(coordI a,coordI b){
	a=a-b;
	int tmp=0;
	if(a.y<0) {a.y=-a.y;tmp|=0b001;}
	if(a.x<0) {a.x=-a.x;tmp|=0b010;}
	if(a.y>a.x) {swap(a.y,a.x);tmp|=0b100;}
	int m=0;int y=0;
	coordI p;
	uint8_t r,g,bl;
	for(int x=0;x<=a.x;x++){
		p={x,y};
		if(tmp&0b100) swap(p.x,p.y);
		if(tmp&0b010) p.x=-p.x;
		if(tmp&0b001) p.y=-p.y;
		p=p+b;
		
		// Check bounds
		if (p.x < 0 || p.x >= W || p.y < 0 || p.y >= H) continue;
		
		// Get pixel color from pen
		pixel px = pen(mem2abs({p.x,p.y}));
		
		// Determine if we should draw (solid) or erase (empty) based on alpha
		// If alpha is 0 (fully transparent), erase the pixel
		// Otherwise, draw as solid pixel
		if (px.a < 0.001f) { // Alpha close to 0 means erase
			cv[p.x][p.y] = 0; // Mark as empty
		} else {
			cv[p.x][p.y] = 1; // Mark as solid
		}
		fi[p.x][p.y] = Complex(0.0, 0.0);
		
		// Update rendering arrays (will be overridden by magic_mapToCvs but kept for immediate visual feedback)
		cvs_r[p.x][p.y] = px.r;
		cvs_g[p.x][p.y] = px.g;
		cvs_b[p.x][p.y] = px.b;
		cvs_a[p.x][p.y] = px.a;
		cvs_lr[p.x][p.y] = px.lr;
		cvs_lg[p.x][p.y] = px.lg;
		cvs_lb[p.x][p.y] = px.lb;
		sample_pix(p, r, g, bl);
		bk[p.x][p.y][0]=r;bk[p.x][p.y][1]=g;bk[p.x][p.y][2]=bl;
		m+=a.y;if(m>=a.x){y++;m-=a.x;}
	}
	flag_gpu_recompute = true;
}

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 position;
out vec2 texCoords;
void main() {
	gl_Position = vec4(position, 0.0, 1.0);
	// 原始映射: texCoords = position * 0.5 + 0.5;
	// 翻转 Y: texCoords.y = 1.0 - (position.y * 0.5 + 0.5) = 0.5 - position.y * 0.5
	texCoords = vec2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
}
)";

const char* fragmentShaderSource = R"(
									  #version 330 core
									  in vec2 texCoords;
									  out vec4 fragColor;
									  uniform sampler2D screenTexture;
									  uniform vec3 cameraParams; // x: cam.x, y: cam.y, z: cam.sk
									  uniform vec2 screenSize;   // x: scrW, y: scrH
									  uniform vec2 textureSize;  // x: W, y: H
									  
									  void main() {
										  vec2 screenPos = texCoords * screenSize;
										  vec2 absPos;
										  absPos.x = cameraParams.z * (screenPos.x / screenSize.x - 0.5) + cameraParams.x;
										  absPos.y = cameraParams.z * (0.5 - screenPos.y / screenSize.y) + cameraParams.y;
										  vec2 memPos = absPos * textureSize;
										  vec2 texCoord = memPos / textureSize;
										  
										  if (texCoord.x < 0.0 || texCoord.x > 1.0 || texCoord.y < 0.0 || texCoord.y > 1.0) {
											  fragColor = vec4(0.0, 0.0, 0.0, 1.0);
										  } else {
											  fragColor = texture(screenTexture, texCoord);
}


 									  }
 									  )";

void loadConfig() {
    // Try to load configuration from cfg.txt
    std::ifstream fin("cfg.txt");
    if (!fin.is_open()) {
        std::cout << "Config file cfg.txt not found, using default colors." << std::endl;
        return;
    }
    
    int cfgW, cfgH;
    fin >> cfgW >> cfgH;
    std::cout << "Config read W=" << cfgW << ", H=" << cfgH << std::endl;
    
    // Read font line
    std::string fontName;
    fin.ignore(); // skip newline after dimensions
    std::getline(fin, fontName); // read font line
    std::cout << "Font: " << fontName << std::endl;
    
    // Read environment light color (background color)
    fin >> std::hex >> color_cfg[0];
    std::cout << "Environment light color: 0x" << std::hex << color_cfg[0] << std::dec << std::endl;
    
    // Read font color
    fin >> std::hex >> color_cfg[3];
    std::cout << "Font color: 0x" << std::hex << color_cfg[3] << std::dec << std::endl;
    
    // Read global transparency
    float cfgAlpha;
    fin >> cfgAlpha;
    if (cfgAlpha >= 0.0f && cfgAlpha <= 1.0f) {
        globalAlpha = cfgAlpha;
        std::cout << "Global alpha: " << globalAlpha << std::endl;
    } else {
        std::cout << "Invalid alpha value in config, using default: " << globalAlpha << std::endl;
    }
    
    // Disable sun lighting as requested
    sunsc = 0.0f;
    std::cout << "Sun lighting disabled (sunsc = 0)" << std::endl;
    
    fin.close();
    
    // Set unused colors to default values
    color_cfg[1] = 0xFFFF0000; // real part (red) - not used
    color_cfg[2] = 0xFF0000FF; // imaginary part (blue) - not used
    
    // Update color_cfgD array
    for (int i = 0; i < 4; i++) {
        color_cfgD[i][0] = (color_cfg[i] >> 24) & 0xFF;
        color_cfgD[i][1] = (color_cfg[i] >> 16) & 0xFF;
        color_cfgD[i][2] = (color_cfg[i] >> 8) & 0xFF;
        color_cfgD[i][3] = color_cfg[i] & 0xFF;
        for (int j = 0; j < 4; j++) {
            color_cfgD[i][j] /= 255.0;
        }
    }
}

const char* computeShaderSource = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) uniform image2D outputImage;

const int W = 128;
const int H = 128;
const float eps_l = 1e-3;
const float eps = 1e-6;
const int R = 128; // max(W, H)
const float invR = 1.0 / float(R);
const float sunsc = 5e-3;
const vec2 sundr = vec2(0.70710678, 0.70710678); // sqrt(0.5)
const vec3 wallcl = vec3(0.2, 0.2, 0.2);
const vec3 suncl = vec3(10.0, 10.0, 10.0);

layout(std430, binding = 1) buffer CVSBuffer {
    float cvs_r[W*H];
    float cvs_g[W*H];
    float cvs_b[W*H];
    float cvs_a[W*H];
    float cvs_lr[W*H];
    float cvs_lg[W*H];
    float cvs_lb[W*H];
};

layout(std430, binding = 2) buffer DirectionBuffer {
    int dirData[]; // 每5个元素一组: dx, dy, octant, absDx, absDy
};

uniform int totalSamples;

// 光线追踪函数，模拟 ray_optimized
vec3 traceRay(ivec2 o, int dirIndex) {
    int base = dirIndex * 5;
    int dx = dirData[base];
    int dy = dirData[base + 1];
    int octant = dirData[base + 2];
    int absDx = dirData[base + 3];
    int absDy = dirData[base + 4];
    
    vec2 dr = vec2(float(dx) * invR, float(dy) * invR);
    
    int m = 0;
    int y = 0;
    ivec2 p;
    vec3 lt = vec3(1.0, 1.0, 1.0);
    
    for (int x = 0; ; x++) {
        // 安全限制，防止无限循环
        if (x > 2 * R) return lt;
        // 根据八分域变换计算实际坐标
        p.x = x;
        p.y = y;
        
        // 应用八分域变换
        if ((octant & 4) != 0) { // 0b100
            int tmp = p.x;
            p.x = p.y;
            p.y = tmp;
        }
        if ((octant & 2) != 0) p.x = -p.x; // 0b010
        if ((octant & 1) != 0) p.y = -p.y; // 0b001
        
        p = p + o;
        
        if (p.x < 0 || p.x >= W || p.y < 0 || p.y >= H) {
            float dotp = 1.0 - (sundr.x * dr.x + sundr.y * dr.y);
            if (dotp <= sunsc) {
                lt *= suncl;
                return lt;
            }
            lt *= wallcl;
            return lt;
        }
        if (lt.r < eps_l && lt.g < eps_l && lt.b < eps_l) return lt;
        
        int idx = p.y * W + p.x;
        float alpha = cvs_a[idx];
        if (1.0 - alpha < eps_l) {
            lt *= vec3(cvs_lr[idx], cvs_lg[idx], cvs_lb[idx]);
            return lt;
        }
        m += absDy;
        if (m >= absDx) { y++; m -= absDx; }
    }
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= W || pixel.y >= H) return;
    
    int idx = pixel.y * W + pixel.x;
    float alpha = cvs_a[idx];
    
    vec3 color;
    if (1.0 - alpha < eps_l) {
        // 不透明像素，直接使用颜色
        color = vec3(cvs_r[idx], cvs_g[idx], cvs_b[idx]);
    } else {
        // 透明像素：对所有预计算方向进行光线追踪并累加
        vec3 accum = vec3(0.0);
        for (int i = 0; i < totalSamples; i++) {
            accum += traceRay(pixel, i);
        }
        float invTotal = 1.0 / float(totalSamples);
        color = accum * invTotal;
        // 颜色裁剪，防止过饱和
        float over = max(max(color.r, color.g), color.b);
        if (over > 1.0) color /= over;
        color = max(color, vec3(0.0));
    }
    
    imageStore(outputImage, pixel, vec4(color, 1.0));
}
)";

void initGL(){
	int success;
	char infoLog[512];

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if(!success){
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << endl;
	}

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if(!success){
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << endl;
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if(!success){
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	float vertices[] = {
		-1.0f, -1.0f,
		 1.0f, -1.0f,
		-1.0f,  1.0f,
		 1.0f,  1.0f
	};

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	// 分配纹理存储空间 (RGBA8格式，计算着色器将写入)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	// 创建计算着色器
	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(computeShader, 1, &computeShaderSource, NULL);
	glCompileShader(computeShader);
	glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
	if(!success){
		glGetShaderInfoLog(computeShader, 512, NULL, infoLog);
		cerr << "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n" << infoLog << endl;
	}

	computeProgram = glCreateProgram();
	glAttachShader(computeProgram, computeShader);
	glLinkProgram(computeProgram);
	glGetProgramiv(computeProgram, GL_LINK_STATUS, &success);
	if(!success){
		glGetProgramInfoLog(computeProgram, 512, NULL, infoLog);
		cerr << "ERROR::COMPUTE_PROGRAM::LINKING_FAILED\n" << infoLog << endl;
	}
  glDeleteShader(computeShader);
  if (success) {
    std::cout << "Compute shader program linked successfully (GPU acceleration enabled)." << std::endl;
  }

	// 创建SSBO用于CVS数据
	glGenBuffers(1, &cvsSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cvsSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, W * H * 7 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	// 创建SSBO用于预计算的方向数据（稍后在init中填充）
	glGenBuffers(1, &dirSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirSSBO);
	// 先分配空数据，init中会重新分配
	glBufferData(GL_SHADER_STORAGE_BUFFER, 0, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void updateTexture(){
	glBindTexture(GL_TEXTURE_2D, textureID);
	if (computeProgram == 0) {
		// CPU计算路径：上传bk数据，转换为RGBA格式
		std::vector<uint8_t> rgba(W * H * 4);
		for (int y = 0; y < H; ++y) {
			for (int x = 0; x < W; ++x) {
				int idx = (y * W + x) * 4;
				rgba[idx + 0] = bk[x][y][0];
				rgba[idx + 1] = bk[x][y][1];
				rgba[idx + 2] = bk[x][y][2];
				rgba[idx + 3] = 255; // Alpha通道
			}
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
	}
	// GPU计算路径：纹理已由计算着色器填充，无需上传
}

void renderQuad(){
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(shaderProgram);
	// 设置相机uniform
	GLint cameraParamsLoc = glGetUniformLocation(shaderProgram, "cameraParams");
	GLint screenSizeLoc = glGetUniformLocation(shaderProgram, "screenSize");
	GLint textureSizeLoc = glGetUniformLocation(shaderProgram, "textureSize");
	if (cameraParamsLoc != -1) {
		glUniform3f(cameraParamsLoc, cam.x, cam.y, cam.sk);
	}
	if (screenSizeLoc != -1) {
		glUniform2f(screenSizeLoc, scrW, scrH);
	}
	if (textureSizeLoc != -1) {
		glUniform2f(textureSizeLoc, W, H);
	}
	
	glBindVertexArray(vao);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glfwSwapBuffers(window);
}

void cleanupGL(){
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glDeleteTextures(1, &textureID);
	glDeleteProgram(shaderProgram);
	if (computeProgram) {
		glDeleteProgram(computeProgram);
		computeProgram = 0;
	}
  if (cvsSSBO) {
    glDeleteBuffers(1, &cvsSSBO);
    cvsSSBO = 0;
  }
  if (dirSSBO) {
    glDeleteBuffers(1, &dirSSBO);
    dirSSBO = 0;
  }
}

// ============================================================================
// Magic Simulation Functions (Placeholder - to be implemented from magic project)
// ============================================================================

void magic_init() {
    // Initialize magic simulation
    // TODO: Load config from cfg.txt
    // For now, set default values
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            cv[x][y] = 0;
            fi[x][y] = Complex(0, 0);
            cv_r[x][y] = 0;
            cv_s[x][y] = 0;
        }
    }
    magic_hand = Complex(0, 0);
    magic_hand_c = coordI(0, 0);
    magic_hand_r = 0;
}

void magic_update() {
    // Update magic simulation (one step)
    magic_parsing();
    magic_conduct();
}

void magic_parsing() {
    // Parse exponential and logarithmic sections (from magic project)
    coordI i;
    for (i.y = 0; i.y < H - 5; i.y++) {
        for (i.x = 0; i.x < W - 5; i.x++) {
            for (int j = 0; j < 8; j++)
                magic_reco_ln(i.x, i.y, j);
            for (int j = 0; j < 4; j++)
                magic_reco_exp(i.x, i.y, j);
        }
    }
}

void magic_conduct() {
    // Conduct signals (from magic project)
    // Clear field and temp array
    memset(fi, 0, sizeof(fi));
    memset(cv_s, 0, sizeof(cv_s));
    int cc = 0;
    coordI p;
    while (sprk.size()) {
        std::queue<coordI> q;
        q.push(sprk.front());
        sprk.pop();
        cc++;
        while (q.size()) {
            p = q.front();
            q.pop();
            // Right
            if (cv[p.x + 1][p.y] && cv[p.x + 2][p.y] &&
                cv[p.x + 1][p.y + 1] && cv[p.x + 1][p.y - 1] &&
                !cv[p.x][p.y + 1] && !cv[p.x][p.y - 1] &&
                !cv[p.x + 2][p.y + 1] && !cv[p.x + 2][p.y - 1]) {
                if (cv_s[p.x + 2][p.y] != cc) {
                    q.push(coordI(p.x + 2, p.y));
                    fi[p.x + 2][p.y] += sprk_f.front();
                    fi[p.x + 1][p.y] = fi[p.x + 2][p.y];
                    cv_s[p.x + 2][p.y] = cc;
                }
            } else {
                if (cv[p.x + 1][p.y] && cv_s[p.x + 1][p.y] != cc) {
                    q.push(coordI(p.x + 1, p.y));
                    fi[p.x + 1][p.y] += sprk_f.front();
                    cv_s[p.x + 1][p.y] = cc;
                }
            }
            // Left
            if (cv[p.x - 1][p.y] && cv[p.x - 2][p.y] &&
                cv[p.x - 1][p.y + 1] && cv[p.x - 1][p.y - 1] &&
                !cv[p.x][p.y + 1] && !cv[p.x][p.y - 1] &&
                !cv[p.x - 2][p.y + 1] && !cv[p.x - 2][p.y - 1]) {
                if (cv_s[p.x - 2][p.y] != cc) {
                    q.push(coordI(p.x - 2, p.y));
                    fi[p.x - 2][p.y] += sprk_f.front();
                    fi[p.x - 1][p.y] = fi[p.x - 2][p.y];
                    cv_s[p.x - 2][p.y] = cc;
                }
            } else {
                if (cv[p.x - 1][p.y] && cv_s[p.x - 1][p.y] != cc) {
                    q.push(coordI(p.x - 1, p.y));
                    fi[p.x - 1][p.y] += sprk_f.front();
                    cv_s[p.x - 1][p.y] = cc;
                }
            }
            // Up
            if (cv[p.x][p.y + 1] && cv[p.x][p.y + 2] &&
                cv[p.x + 1][p.y + 1] && cv[p.x - 1][p.y + 1] &&
                !cv[p.x + 1][p.y] && !cv[p.x - 1][p.y] &&
                !cv[p.x + 1][p.y + 2] && !cv[p.x - 1][p.y + 2]) {
                if (cv_s[p.x][p.y + 2] != cc) {
                    q.push(coordI(p.x, p.y + 2));
                    fi[p.x][p.y + 2] += sprk_f.front();
                    fi[p.x][p.y + 1] = fi[p.x][p.y + 2];
                    cv_s[p.x][p.y + 2] = cc;
                }
            } else {
                if (cv[p.x][p.y + 1] && cv_s[p.x][p.y + 1] != cc) {
                    q.push(coordI(p.x, p.y + 1));
                    fi[p.x][p.y + 1] += sprk_f.front();
                    cv_s[p.x][p.y + 1] = cc;
                }
            }
            // Down
            if (cv[p.x][p.y - 1] && cv[p.x][p.y - 2] &&
                cv[p.x + 1][p.y - 1] && cv[p.x - 1][p.y - 1] &&
                !cv[p.x + 1][p.y] && !cv[p.x - 1][p.y] &&
                !cv[p.x + 1][p.y - 2] && !cv[p.x - 1][p.y - 2]) {
                if (cv_s[p.x][p.y - 2] != cc) {
                    q.push(coordI(p.x, p.y - 2));
                    fi[p.x][p.y - 2] += sprk_f.front();
                    fi[p.x][p.y - 1] = fi[p.x][p.y - 2];
                    cv_s[p.x][p.y - 2] = cc;
                }
            } else {
                if (cv[p.x][p.y - 1] && cv_s[p.x][p.y - 1] != cc) {
                    q.push(coordI(p.x, p.y - 1));
                    fi[p.x][p.y - 1] += sprk_f.front();
                    cv_s[p.x][p.y - 1] = cc;
                }
            }
        }
        sprk_f.pop();
    }
}

void magic_reco_ln(int x, int y, int dir) {
    // Exact implementation from magic project's reco_ln
    switch (dir) {
        case 0:
            if (cv[x + 0][y + 0]) break;
            if (!cv[x + 1][y + 0]) break;
            if (cv[x + 2][y + 0]) break;
            if (!cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (cv[x + 3][y + 2]) break;
            if (cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 4][y + 3]) break;
            if (cv[x + 1][y + 4]) break;
            if (cv[x + 2][y + 4]) break;
            if (cv[x + 3][y + 4]) break;

            sprk_f.push(std::log(fi[x + 2][y + 1]));
            sprk.push(coordI(x + 2, y + 3));
            break;
        case 1:
            if (cv[x + 3][y + 0]) break;
            if (!cv[x + 1][y + 1]) break;
            if (cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 4][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (!cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 4][y + 3]) break;
            if (cv[x + 0][y + 4]) break;
            if (!cv[x + 1][y + 4]) break;
            if (cv[x + 2][y + 4]) break;
            if (cv[x + 3][y + 4]) break;

            sprk_f.push(std::log(fi[x + 1][y + 2]));
            sprk.push(coordI(x + 3, y + 2));
            break;
        case 2:
            if (cv[x + 1][y + 0]) break;
            if (cv[x + 2][y + 0]) break;
            if (cv[x + 3][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 4][y + 1]) break;
            if (cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (!cv[x + 4][y + 3]) break;
            if (cv[x + 2][y + 4]) break;
            if (!cv[x + 3][y + 4]) break;
            if (cv[x + 4][y + 4]) break;

            sprk_f.push(std::log(fi[x + 2][y + 3]));
            sprk.push(coordI(x + 2, y + 1));
            break;
        case 3:
            if (cv[x + 1][y + 0]) break;
            if (cv[x + 2][y + 0]) break;
            if (!cv[x + 3][y + 0]) break;
            if (cv[x + 4][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (!cv[x + 4][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 1][y + 4]) break;

            sprk_f.push(std::log(fi[x + 3][y + 2]));
            sprk.push(coordI(x + 1, y + 2));
            break;
        case 4:
            if (cv[x + 2][y + 0]) break;
            if (!cv[x + 3][y + 0]) break;
            if (cv[x + 4][y + 0]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (!cv[x + 4][y + 1]) break;
            if (cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 4][y + 3]) break;
            if (cv[x + 1][y + 4]) break;
            if (cv[x + 2][y + 4]) break;
            if (cv[x + 3][y + 4]) break;

            sprk_f.push(std::log(fi[x + 2][y + 1]));
            sprk.push(coordI(x + 2, y + 3));
            break;
        case 5:
            if (cv[x + 1][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (!cv[x + 4][y + 3]) break;
            if (cv[x + 1][y + 4]) break;
            if (cv[x + 2][y + 4]) break;
            if (!cv[x + 3][y + 4]) break;
            if (cv[x + 4][y + 4]) break;

            sprk_f.push(std::log(fi[x + 3][y + 2]));
            sprk.push(coordI(x + 1, y + 2));
            break;
        case 6:
            if (cv[x + 1][y + 0]) break;
            if (cv[x + 2][y + 0]) break;
            if (cv[x + 3][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 4][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (cv[x + 3][y + 2]) break;
            if (!cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 0][y + 4]) break;
            if (!cv[x + 1][y + 4]) break;
            if (cv[x + 2][y + 4]) break;

            sprk_f.push(std::log(fi[x + 2][y + 3]));
            sprk.push(coordI(x + 2, y + 1));
            break;
        case 7:
            if (cv[x + 0][y + 0]) break;
            if (!cv[x + 1][y + 0]) break;
            if (cv[x + 2][y + 0]) break;
            if (cv[x + 3][y + 0]) break;
            if (!cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 4][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (!cv[x + 1][y + 3]) break;
            if (cv[x + 2][y + 3]) break;
            if (!cv[x + 3][y + 3]) break;
            if (cv[x + 4][y + 3]) break;
            if (cv[x + 3][y + 4]) break;

            sprk_f.push(std::log(fi[x + 1][y + 2]));
            sprk.push(coordI(x + 3, y + 2));
            break;
        default:
            break;
    }
    return;
}
void magic_reco_exp(int x, int y, int dir) {
    // Exact implementation from magic project's reco_exp
    switch (dir) {
        case 0:
            if (cv[x + 1][y + 0]) break;
            if (!cv[x + 2][y + 0]) break;
            if (cv[x + 3][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (!cv[x + 3][y + 1]) break;
            if (cv[x + 4][y + 1]) break;
            if (cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (cv[x + 3][y + 2]) break;
            if (!cv[x + 2][y + 3]) break;

            if (!(cv[x + 2][y + 4] || cv[x + 1][y + 3] || cv[x + 3][y + 3])) {
                sprk_f.push(Complex(-1, 0));
                sprk.push(coordI(x + 2, y + 1));
            } else {
                sprk_f.push(std::exp(fi[x + 2][y + 3]));
                sprk.push(coordI(x + 2, y + 1));
            }
            break;
        case 1:
            if (cv[x + 1][y + 0]) break;
            if (cv[x + 0][y + 1]) break;
            if (!cv[x + 1][y + 1]) break;
            if (cv[x + 2][y + 1]) break;
            if (!cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 0][y + 3]) break;
            if (!cv[x + 1][y + 3]) break;
            if (cv[x + 2][y + 3]) break;
            if (cv[x + 1][y + 4]) break;

            if (!(cv[x + 4][y + 2] || cv[x + 3][y + 1] || cv[x + 3][y + 3])) {
                sprk_f.push(Complex(-1, 0));
                sprk.push(coordI(x + 1, y + 2));
            } else {
                sprk_f.push(std::exp(fi[x + 3][y + 2]));
                sprk.push(coordI(x + 1, y + 2));
            }
            break;
        case 2:
            if (!cv[x + 2][y + 0]) break;
            if (cv[x + 1][y + 1]) break;
            if (cv[x + 2][y + 1]) break;
            if (cv[x + 3][y + 1]) break;
            if (cv[x + 0][y + 2]) break;
            if (!cv[x + 1][y + 2]) break;
            if (!cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 4][y + 2]) break;
            if (cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (cv[x + 3][y + 3]) break;

            if (!(cv[x + 2][y - 1] || cv[x + 1][y + 0] || cv[x + 3][y + 0])) {
                sprk_f.push(Complex(-1, 0));
                sprk.push(coordI(x + 2, y + 2));
            } else {
                sprk_f.push(std::exp(fi[x + 2][y + 0]));
                sprk.push(coordI(x + 2, y + 2));
            }
            break;
        case 3:
            if (cv[x + 2][y + 0]) break;
            if (cv[x + 1][y + 1]) break;
            if (!cv[x + 2][y + 1]) break;
            if (cv[x + 3][y + 1]) break;
            if (!cv[x + 0][y + 2]) break;
            if (cv[x + 1][y + 2]) break;
            if (!cv[x + 2][y + 2]) break;
            if (!cv[x + 3][y + 2]) break;
            if (cv[x + 1][y + 3]) break;
            if (!cv[x + 2][y + 3]) break;
            if (cv[x + 3][y + 3]) break;
            if (cv[x + 2][y + 4]) break;

            if (!(cv[x - 1][y + 2] || cv[x + 0][y + 1] || cv[x + 0][y + 3])) {
                sprk_f.push(Complex(-1, 0));
                sprk.push(coordI(x + 2, y + 2));
            } else {
                sprk_f.push(std::exp(fi[x + 0][y + 2]));
                sprk.push(coordI(x + 2, y + 2));
            }
            break;
        default:
            break;
    }
    return;
}

Complex magic_getCellValue(coordI cell) {
    if (cell.x >= 0 && cell.x < W && cell.y >= 0 && cell.y < H) {
        return fi[cell.x][cell.y];
    }
    return Complex(0, 0);
}

void magic_setMagicHand(const Complex& value) {
    magic_hand = value;
}

uint32_t magic_c2col(const Complex& c) {
    // Use the same mapping as magic project's c2col1
    return magic_c2col1(c);
}

uint32_t magic_c2col1(const Complex& c) {
    // Exact implementation from magic project's c2col1
    float xabs = 1.0f - powf(0.5f, static_cast<float>((std::abs(c) - 1.0) * ARGBIAS2 + 1.0));
    float xarg = static_cast<float>(std::arg(c) * PI_1 + ARGBIAS1);
    if (xarg < 0) xarg += 2.0f;
    // hsl2rgb expects hue in degrees (0-360), saturation=1.0, lightness=xabs
    return hsl2rgb(xarg * 180.0f, 1.0f, xabs);
}

void magic_mapToCvs() {
    // Map magic field to cvs arrays for rendering
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Convert complex to color
            uint32_t argb = magic_c2col(fi[x][y]);
            uint8_t a = (argb >> 24) & 0xFF;
            uint8_t r = (argb >> 16) & 0xFF;
            uint8_t g = (argb >> 8) & 0xFF;
            uint8_t b = argb & 0xFF;
            
            // Apply alpha based on cell state
            float alpha;
            if (cv[x][y] == 0) {
                // Empty cell: fully transparent
                alpha = 0.0f;
            } else {
                // Solid cell: fully opaque (to avoid ray-tracing for simple drawn lines)
                // This matches original magic behavior: solid pixels show color directly
                alpha = 1.0f;
            }
            
            // Store in cvs arrays (normalized to 0-1)
            cvs_r[x][y] = r / 255.0f;
            cvs_g[x][y] = g / 255.0f;
            cvs_b[x][y] = b / 255.0f;
            cvs_a[x][y] = alpha;
            
            // Emission (if enabled)
            if (useEmission) {
                // Simple emission based on magnitude
                double mag = std::abs(fi[x][y]);
                float emission = static_cast<float>(mag / (1.0 + mag));
                cvs_lr[x][y] = cvs_r[x][y] * emission;
                cvs_lg[x][y] = cvs_g[x][y] * emission;
                cvs_lb[x][y] = cvs_b[x][y] * emission;
            } else {
                cvs_lr[x][y] = 0.0f;
                cvs_lg[x][y] = 0.0f;
                cvs_lb[x][y] = 0.0f;
            }
        }
    }
    flag_gpu_recompute = true;
}

