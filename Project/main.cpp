#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#define alpha 0.1
using namespace sycl;

//#激活函数sigmoid:logistic
float f(float val) 
{ 
    return (1/(1+exp(-val))); 
}

float rand_() 
{ 
    return  rand() / float(RAND_MAX); 
}

std::vector<float> mulmatrix(const std::vector<float>& matrixA, 
        const std::vector<float>& matrixB, int rowsA, int colsA, int colsB,queue &q) 
{
    std::vector<float> result(rowsA * colsB, 0.f);
    buffer<float, 2> bufferA(matrixA.data(), range<2>(rowsA, colsA));
    buffer<float, 2> bufferB(matrixB.data(), range<2>(colsA, colsB));
    buffer<float, 2> bufferResult(result.data(), range<2>(rowsA, colsB));

    q.submit([&](handler& h) 
    {
        auto accessorA = bufferA.get_access<access::mode::read>(h);
        auto accessorB = bufferB.get_access<access::mode::read>(h);
        auto accessorResult = bufferResult.get_access<access::mode::write>(h);

        range<2> global_size(rowsA, colsB);
        range<2> work_group_size(rowsA, colsB);

        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item)
        {
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            float temp = 0.f;
            for (int k = 0; k < colsA; k++) 
            {
                temp += accessorA[{static_cast<size_t>(i), static_cast<size_t>(k)}] * 
                        accessorB[{static_cast<size_t>(k), static_cast<size_t>(j)}];

            }
            accessorResult[{static_cast<size_t>(i), static_cast<size_t>(j)}] = temp;

        });
    }).wait();

    //#std::cout << "\n----FINISH MULTIPLY----\n";
    return result;
}

std::vector<float> T(const std::vector<float>& matrix,int row, int col)
{
    std::vector<float> t(col*row);
    //t[i,j]=matrix[j,i]
    for(int i=0;i<col;i++)
    for(int j=0;j<row;j++) t[i*row+j]=matrix[j*col+i];
    return t;
}

std::vector<float> dotmatrix(const std::vector<float>& matrixA, 
            const std::vector<float>& matrixB, int rows, int cols,queue &q)
{
    std::vector<float> result(rows * cols, 0.f);
    buffer<float, 2> bufferA(matrixA.data(), range<2>(rows, cols));
    buffer<float, 2> bufferB(matrixB.data(), range<2>(cols, cols));
    buffer<float, 2> bufferResult(result.data(), range<2>(rows, cols));
    q.submit([&](handler& h) 
    {
        auto accessorA = bufferA.get_access<access::mode::read>(h);
        auto accessorB = bufferB.get_access<access::mode::read>(h);
        auto accessorResult = bufferResult.get_access<access::mode::write>(h);

        range<2> global_size(rows, cols);
        range<2> work_group_size(rows, cols);

        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item)
        {
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            float temp = accessorA[{static_cast<size_t>(i), static_cast<size_t>(j)}] * accessorB[{static_cast<size_t>(i), static_cast<size_t>(j)}];
            accessorResult[{static_cast<size_t>(i), static_cast<size_t>(j)}] = temp;
        });
    }).wait();

    return result;
}

//#.csv文件已经归一化
std::pair<std::vector<std::vector<float>> , std::vector<std::vector<float>> > inputprocess()
{
    std::vector<std::vector<float>> wine;//#1595个1*11矩阵
    std::vector<float> inputMatrix;//#11*11矩阵
    std::vector<std::vector<float>>retmatrix;//#145个11*11矩阵
    std::vector<float>labels;//#1585个label
    std::vector<float>label;//#1*11矩阵
    std::vector<std::vector<float>>retlabel;//#145个11*1矩阵

    size_t num=0; //#就是1595
    size_t cnt=0; //#[0,11]
    
    std::ifstream fp("redwine.csv"); 
    std::string line;
    std::getline(fp,line); //#第一行是列名，不做处理
    while (std::getline(fp,line))
    { 
        std::vector<float> data_line;
        std::string number;
        std::istringstream readstr(line); 
        for(int j = 0;j < 11;j++)//#一组12个数字里前11个是特征
        { 
            getline(readstr,number,','); 
            float now=atof(number.c_str());//#要字符串转float

            data_line.push_back(now); 
        }
        getline(readstr,number,',');//#第12个数字是label
        float y=atof(number.c_str());

        wine.push_back(data_line); 
        labels.push_back(y);
        inputMatrix.insert(inputMatrix.end(), data_line.begin(), data_line.end());
        label.push_back(y);
        num++;
        cnt++;
        if(cnt==11)
        {
            retmatrix.push_back(inputMatrix);
            retlabel.push_back(label);
            inputMatrix.clear();
            label.clear();
            cnt=0;
        }
    }
    //#std::cout<<num<<"  "<<(num/11)<<"\n";

    return std::pair<std::vector<std::vector<float>>,
    std::vector<std::vector<float>>>(retmatrix,retlabel);
}

double Loss(const std::vector<float>& output, const std::vector<float>& label,int num)//#1*11,用的MSE
{
    double sum = 0;
    for(int i=0;i<num;i++) 
    sum+=pow((output[i]-label[i]),2);
    return sum/(2*num);
}

struct layer
{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> fy;
    std::vector<float> w;
    std::vector<float> b;
    int w1,w2,b1,b2;
    std::vector<float> o,w_grad;//#误差项 W转置*误差项
    int o1,o2;

    layer(int r,int c)
    {
        w.resize(r*c);
        b.resize(r*11);
        x.resize(c*11);
        y.resize(r*11); 
        fy.resize(r*11); 
        o.resize(r*11); 
        w_grad.resize(r*11);
        w1=r; 
        w2=c;
        b1=r; 
        b2=11;
        for(int i=0;i<w1;i++)
            for(int j=0;j<w2;j++) 
                w[i*w2+j]=1.0f;//#rand_();
        for(int i=0;i<b1;i++)
            for(int j=0;j<b2;j++) 
                b[i*b2+j]=1.0f;//#rand_();
    }

    std::vector<float> forward(const std::vector<float>& input,queue& q)//#输入矩阵X 
    {
        x=input;
        y=mulmatrix(w,input,w1,w2,b2,q);
        for(int i=0;i<w1;i++)
        for(int j=0;j<b2;j++) fy[i*b2+j]=f(y[i*b2+j]);

        return fy;
    }

    std::vector<float> backward(const std::vector<float>& w_g,queue& q)//#下一层传回的w_grad(i+1)
    {
        
        std::vector<float> dy(b1*b2);
        for(int i=0;i<b1;i++)
        for(int j=0;j<b2;j++)
        {
            int pos=i*b2+j;
            dy[pos]=fy[pos]*(1-fy[pos]);//#f'(y)=f(y)*(1-f(y))
        }
        o=dotmatrix(dy,w_g,b1,b2,q);
        w_grad=mulmatrix(T(w,w1,w2),o,w2,w1,11,q);

        return w_grad;
    }

    //#最后一层o(i)
    std::vector<float> back(const std::vector<float>& label,queue& q)
    {
        for(int i=0;i<b2;i++) 
            o[i]=(y[i]-label[i])/b2;
        w_grad=mulmatrix(T(w,w1,w2),o,w2,w1,b2,q);

        return w_grad;
    }

    void update(queue &q)
    {
        //#W(i)=W(i)-alpha*o(i)*X(i)转置
        //#b(i)=b(i)-alpha*o(i)
        std::vector<float> tmp=mulmatrix(o,T(x,w2,b2),b1,b2,w2,q);
        for(int i=0;i<w1;i++)
        for(int j=0;j<w2;j++)
        {
            int pos=i*w2+j;
            w[pos]=w[pos]-(alpha*tmp[pos]);
        }
        for(int i=0;i<b1;i++)
        for(int j=0;j<b2;j++) 
        {
            int pos=i*b2+j;
            b[pos]=b[pos]-(alpha*o[pos]);
        } 
    }
};

int main()
{
    std::pair<std::vector<std::vector<float>>,std::vector<std::vector<float>>> ret=inputprocess();
    std::vector<std::vector<float>> matrix_x = ret.first;//#145个11*11特征矩阵
    std::vector<std::vector<float>> matrix_y = ret.second;//#145个1*11label矩阵
    queue q;
    layer layer1(9,11);
    layer layer2(3,9);
    layer layer3(1,3);

        for(int i=0;i<50;i++)
        {
            double avrloss=0;
            for(int j=0;j<120;j++)//#有145组数据，后25组测试使用
            {
                std::vector<float> input=matrix_x[j];
                std::vector<float> label=matrix_y[j];
                std::vector<float> output=layer3.forward(
                    layer2.forward(
                        layer1.forward(input,q),q),q);
                for(int i=0;i<11;i++) 
                output[i]=-std::log(1/output[i]-1);
                double loss=Loss(output,label,11);
                avrloss+=loss;
                layer1.backward(
                    layer2.backward(
                        layer3.back(label,q),q),q);
                layer1.update(q);
                layer2.update(q);
                layer3.update(q);
            }
            avrloss/=120; 
            std::cout<<avrloss<<"\n";
        }
        double avl=0;
        for(int j=0;j<25;j++)
        {
            std::vector<float> input=matrix_x[j];
            std::vector<float> label=matrix_y[j];
            std::vector<float> output=layer3.forward(
                layer2.forward(
                    layer1.forward(input,q),q),q);
            for(int i=0;i<11;i++) 
                output[i]=-std::log(1/output[i]-1);
            double loss=Loss(output,label,11);
            avl+=loss;
        }
        avl/=25; 
        std::cout<<"Test:"<<avl<<"\n";
    return 0;
}

