%% 利用群体智能算法进行机器学习超参数优化
% 读取数据
clc;clear                                % 清理界面和变量
data = xlsread('数据模板.xlsx');          % 读取数据
input=data(:,1:end-1);                   % 训练特征（自变量）
output=data(:,end);                      % 输出变量（因变量）
N=length(output);                        % 计算样本量
M=size(input,2);                         % 特征数量

%% 未优化训练SVM预测模型
rng(1);                                     % 固定随机数（为了重现）
SVMModel = fitcsvm(input,output,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');                  % 径向基核训练 SVM 分类器 
CVSVMModel = crossval(SVMModel,'KFold',5);  % 交叉验证 5折SVM 分类器
classLoss = kfoldLoss(CVSVMModel);          % 计算交叉验证误差

%% 调用群体智能算法
n = 30; % 粒子数量
Max_iteration = 10;                                % 迭代的次数
narvs = 1;                                         % 变量个数(核尺度优化)
x_lb=ones(1,narvs);                                % 匹配自变量维度的下届
x_ub=100*ones(1,narvs);                            % 匹配自变量维度的上届
fun = @(x) OBj4(x,input,output);                   % 目标函数
% 粒子群算法优化函数
[gBestScore,gbest,fitnessbest]=IBFO(n,Max_iteration,x_lb,x_ub,narvs,fun); 

%% 绘制收敛曲线
figure(1)
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
semilogy(iter(k),fitnessbest(k),'-*','LineWidth',1)
legend('PSO')
grid off
xlabel('迭代次数')
ylabel('目标函数值')


