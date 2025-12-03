% Genera base de datos sintÃ©tica con puntos calientes siguiendo modelo de
% Stefan-Boltzmann y enfriamiento de Newton
clear;
close all;

ruta='/Users/David/Desktop/matlab_resisto/';

pathdata=[ruta 'volumes/matrices/caminos/CT000001_2023-06-23/']; %Datos
pathread=[ruta 'npy-matlab-master/npy-matlab/']; %lectura funcion readNPY
pathtemplate=[ruta 'volumes/masks/template.npy'];
pathout=[ruta 'data/'];
pathmasks=[ruta 'masks.mat'];
directorio=pwd;
cd(pathread)
template=readNPY(pathtemplate)';
cd(directorio)
tam=[384 288];
load(pathmasks);
% 
D=rdir([pathdata '**/*.npy']); % leemos todas las imágenes
fecha=datetime(D(2).name(end-22:end-4),'InputFormat','yyyy-MM-dd_HH-mm-ss');

%% Crea Máscara
mat_regions={};
num_reg=0;
I=zeros(size(template'));
for i=1:width(masks)

    %lectura de mascaras
    maskvect=masks{2:end,i};
    mask=reshape(maskvect,tam);
    mask= double(mask== 'True');
    
    NumC=1; %Seleccionas el numero de clusters por region
    MaxArea=floor(sum(sum(mask))/NumC);     % Maximum character area for MSER
    MinArea=floor(sum(sum(mask))/(10*NumC));% Minimum area of an item within an image

    mask = mask';
    
    %% Extracting MSER regions
    [regions] = detectMSERFeatures(mask,'RegionAreaRange',[MinArea MaxArea],...
        'ThresholdDelta',2,'MaxAreaVariation',0.25);
    figure;
    imshow(mask); hold on;
    plot(regions,'showPixelList',true,'showEllipses',true);
    
    % remove black regions
    if ~isempty(regions)
        
        regions=remove_MSERneg(mask,regions);
        for k=1:regions.Count
            subidx=regions(k).PixelList;
            IND = sub2ind(size(I),subidx(:,2),subidx(:,1)); % Está al revés
            I(IND)=k+num_reg;
        end
        num_reg=num_reg+regions.Count;
        
        for r=1:regions.Count
            
            mat_regions{end+1}= regions(r).PixelList;
            
        end
    end
end

template=I;
% save('template.mat','template','mat_regions','fecha');

%% Extracting series.. one day...
 
for i=1:numel(D)
    cd(pathread);
    disp(['Extracting figure ' D(i).name ' de ' num2str(numel(D)-1)]) 
    It=readNPY(D(i).name);
    cd(directorio)
    It=reshape(It,tam)';
    for nr=1:num_reg
        mat_sig(i,nr)=mean2(It(I'==nr-1));
    end
    timeday(i)=datetime(D(i).name(end-22:end-4),'InputFormat','yyyy-MM-dd_HH-mm-ss');
    %datetime(Y,M,D,H,MI,S)
end

save('temp_one.mat','mat_sig','timeday');


close all
load('temp_one.mat'); % carga mat_sig y time
load('temperaturasG.mat'); % carga temperaturasG Ambientes
dates=[temperaturasG.FECHA; temperaturasG.FECHA1;temperaturasG.FECHA2];
TsMax=([temperaturasG.TMxima;temperaturasG.TMxima1;temperaturasG.TMxima2]); %Temperaturas ambientes maximas
TsMin=([temperaturasG.TMnima;temperaturasG.TMnima1;temperaturasG.TMnima2]); %Temperaturas ambientes minimas

datap=zeros(size(mat_sig,2),numel(dates)-1,numel(timeday));
alarms=zeros(size(mat_sig,2),numel(dates)-1,numel(timeday));

for numcl=1:size(mat_sig,2)
    
    disp(['Procesando cluster ' num2str(numcl)])
    
    %[v,idx]=max(mat_sig,[],2); %Selecting hottest cluster
    %selec_cluster=mode(idx);
    selec_cluster=numcl;
    %imagesc((I==selec_cluster)'.*template');
    
    %% Fitting
    
    Xtrain=(1:1:numel(timeday))';
    Tmed=double(squeeze(mat_sig(:,selec_cluster)));
    Tmed = medfilt1(Tmed,5);
  
    Numint=5;
    
    N=numel(Xtrain);
    T_op=[];
    for i=1:Numint
        Xtrains=Xtrain((i-1)*N/Numint+1:i*N/Numint);
        Ytrains=Tmed((i-1)*N/Numint+1:i*N/Numint);
        [curve,gof] = fit(Xtrains,Ytrains,"poly2");
        T_op=[T_op; feval(curve,Xtrains)]; % Temperaturas del sistema en un dia àra cada minuto T = Top + error(T amb)
%         subplot(1,2,1)
%         hold on;
%         plot(timeday((i-1)*N/Numint+1:i*N/Numint),feval(curve,Xtrains),timeday((i-1)*N/Numint+1:i*N/Numint),Ytrains);
%         grid on
    end
    
   
    %Computing normalized distribution of errors
    
    error= Tmed-T_op; % Error de la aproximacion
    pd = fitdist(error,'Normal'); % distrbucion gaussiana N((Tmax-Tmin)mu,(Tmax-Tmin)sigma) hipotesis para incluir la dependencia con la T ambiente
    % de la parte dependiente de las condiciones de contorno T amb. Mayor
    % incremento de temperaturas mayor distribucion del error en el modelo.
    %Seleccionando T max y T min del dia
    fechastr=datestr(fecha,'dd-mmmm');
    fecha=datetime([fechastr '-2022']);
    [Tmax,Tmin]=find_temp_table(fecha,temperaturasG);
    % Recuperamos la distribucion normal original
    dispe=Tmax-Tmin;
    sigma_c=pd.sigma/dispe;
    mu_c=pd.mean/dispe;
    %histogram(error);
    %hold on;
    %% Extrapolating the whole year
    
    % Generation
    Tamb=(Tmax+Tmin)/2;
    
    for d=1:numel(dates)-1 % 365 dias del año
 
        % Generamos valores de series en el dia
        Tmaxd=str2num(TsMax(d));
        Tmind=str2num(TsMin(d));
        Tambd=(Tmaxd+Tmind)/2;
        
        %% errores del modelo dependientes de Tamb
        
        dispnew=(Tmaxd-Tmind);
        % Realmente basta con hacer errors=error*dispnew/dispe, asuminedo N(Tmu,Tsig) 
        % Y= (sigma2/sigma1)*X + mu2 - mu1*(sigma2/sigma 1), 
        %donde X->N(mu1,sigma1) y Y-> N(mu2,sigma2)
        errors = normrnd(mu_c*dispnew,...
            sigma_c*dispnew,[numel(T_op),1]);
        %histogram(errors);
        %legend({['Tdisp(t)=' num2str(dispe)],['Tdisp(t+N)=' num2str(dispnew)]})
        %% Estimación T medida dependiente de Tamb
        %  Ley de enfriamiento de newton no se considera y sí la Ley de
        %  Stefan-Boltzmann (emisividad)
        % mat_f=alpha*Q
        % Q=emiss*Boltz(T^4-Tamb^4)
        % por tanto mat_fnew=mat_f+(Tambd^4-T^4ambf)/(alpha*emiss*Boltz)

        %Se asume en los clusters que el Area es un tercio de la masa.
        m = 4.0/1000.0;     % kg
        A = 12.0 / 100.0^2; % Area in m^2
       
        % Parameters Radiation and cooling
        alpha=0.1;    % coeficiente de recepción de Calor cámara
        sigma=5.67e-8;% W/m2-K4; Stefan-Boltzman
        emiss=0.25;   % Emisividad
        %Metales        T [ºC]      emiss
        %Aluminio       170         0,05
        %Acero         -70...700    0,06...0,25
        %Cobre          300..700    0,015...0,025
        %Cobre oxidado  130     0,73
        U = 10.0;           % W/m^2-K Coeficiente de transferencia de calor
        Cp = 0.5 * 1000.0;  % J/kg-K Calor específico
        %Sustancia 	Calor específico (J/kg·K)
        %Acero 	460
        %Aluminio 	880
        %Cobre 	390
        %Estaño 	230
        %Hierro 	450
        %Mercurio 	138
        %Oro 	130
        %Plata 	235
        %Plomo 	130
        %Sodio 	1300
        beta = 0.01;       % W / % heater
      
       % T medida proveniente de la radiación

        T_op_new=((T_op+273.15)+alpha*emiss*sigma*((Tamb+273.15)^4-(Tambd+273.15)^4))-273.15;
        
        %% Datos nuevos T med mas error
        Tmed=errors+T_op_new; % Salvamos todos los clusters, todos las fechas, todos los tiempos
%         subplot(1,2,1);
%         timeday.Year=year(dates(d));
%         timeday.Month=month(dates(d));
%         timeday.Day=day(dates(d));
%         
%        plot(timeday,Tmed);
%        grid on
        %formato CT000001_CA1_2023-06-17_00-03-08
        
        %% Añadimos anomalía sistema térmico completo
        
        palarm=0.1; % probability of alarm in each segment 35 dias al año con alarma
        if rand<palarm
            indt = randi([1 numel(Tmed)]);
            [~,TK]=heatfunction(Tmed(indt),U,m,Cp,A,beta,emiss,sigma); %simula sobre calientamiento
            Decimate=60; % Sampling factor 1 min
            TKres=TK(1:Decimate:end);
            %Tmed=TKres; % aqui se asume que la cámara capta el calentamiento 1 a 1
            Tmednew=(Tmed(indt)+273.15)+alpha*sigma*emiss*(TKres.^4-(Tambd+273.15)^4); % error nulo en comparacion con la anomalia
            %(Tmed(indt)+273.15) hace el papel de Top
            TmednewC=Tmednew-273.15; %Celsius
            idxf=min([indt+numel(TmednewC)-1 numel(Tmed)]);
            Tmed(indt:idxf)=TmednewC(1:idxf-indt+1);
%             subplot(1,2,2);
%             plot(timeday,Tmed);
%             grid on
%             title(['cluster ' num2str(numcl)])
            alarms(numcl,d,indt:idxf)=1; %create alarm variable
        end
        datap(numcl,d,:)=Tmed;
        % Salvamos la serie 
        
    end
    
    
end

save([pathout 'series_alarm/alarms.mat'],'alarms');

%% Salvamos Series
for d=1:numel(dates)-1
    Tmed=zeros(size(mat_sig,2),numel(timeday));
    Tmed=squeeze(datap(:,d,:));
    disp(['Saving CT000001_CA1_'  num2str(dates.Year(d)) '-' num2str(dates.Month(d)) '-' num2str(dates.Day(d))])
    save([pathout 'series_alarm/' 'CT000001_CA1_' num2str(dates.Year(d)) '-' num2str(dates.Month(d)) '-' num2str(dates.Day(d)) '.mat'],'seriest')
end


%% Salvamos secuencia de imagenes
load('template.mat');
for d=1:numel(dates)-1
    disp(['Saving Images for date ' datestr(dates(d))])
    Idates=zeros(size(template));
    
    for t=1:numel(T_op)
        
        datet=D(t+1).name(end-11:end-4);
        
        for k=1:size(mat_sig,2)
            subidx=mat_regions{k};
            IND = sub2ind(size(Idates),subidx(:,2),subidx(:,1));
            
            Idates(IND)=datap(k,d,t);
            
        end
        % Formato CT000001_CA1_2023-06-17_23-42-08
        try
            save([pathout 'images_alarm/' num2str(dates.Month(d)) '/' num2str(dates.Day(d)) '/' 'CT000001_CA1_' num2str(dates.Year(d)) '-' num2str(dates.Month(d)) '-' num2str(dates.Day(d))...
                '_' datet '.mat'],'Idates')
        catch ME
            mkdir([pathout 'images_alarm/' num2str(dates.Month(d)) '/' num2str(dates.Day(d))]);
            save([pathout 'images_alarm/' num2str(dates.Month(d)) '/' num2str(dates.Day(d)) '/' 'CT000001_CA1_' num2str(dates.Year(d)) '-' num2str(dates.Month(d)) '-' num2str(dates.Day(d))...
                '_' datet '.mat'],'Idates')
        end
    end
    
end

%% %%%%% SUBFUNCIONES %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Tmax,Tmin]=find_temp_table(fecha,temperaturasG)

idx = find(contains(temperaturasG.Properties.VariableNames,'FECHA'));
for i=1:numel(idx)
    [idxl ,index] = ismember(fecha,temperaturasG.(idx(i)));
    if idxl
        break
    end
end

Tmax=str2num(temperaturasG.(idx(i)+1)(index));
Tmin=str2num(temperaturasG.(idx(i)+2)(index));

end

function regions=remove_MSERneg(Iin,regions)

idw=false(1,regions.Count);
intensity=[];
IND=[];
for i=1:regions.Count
    subidx=regions(i).PixelList;
    IND = sub2ind(size(Iin),subidx(:,2),subidx(:,1)); % Está al revés
    intensity(i)=sum(Iin(IND));
end
Z=(intensity)>0;
idw(Z>0)=true;
regions=regions(idw);

end

function [time,TK]=heatfunction(T0,U,m,Cp,A,beta,eps,sigma)

%Q = 100.0; % Percent Heater (0-100%)
TK0 = T0 + 273.15; % Initial temperature
n = 60*20;  % Number of min time points (20min)
time = linspace(-n/2+1,n/2-1,n); % Time vector
Q= 100*rectpuls(time,numel(time)/5);

idx=find(Q>0);

[time1,TK1] = ode23(@(t,x)heat(t,x,Q(1),T0,U,m,Cp,A,beta,eps,sigma),time(1:idx(1)-1),TK0); % Integrate ODE
[time2,TK2] = ode23(@(t,x)heat(t,x,Q(idx(1)),T0,U,m,Cp,A,beta,eps,sigma),time(idx(1):idx(end)),TK1(end)); % Integrate ODE
[time3,TK3] = ode23(@(t,x)heat(t,x,Q(idx(end)+1),T0,U,m,Cp,A,beta,eps,sigma),time(idx(end)+1:end),TK2(end)); % Integrate ODE

time=[time1;time2;time3];
TK=[TK1;TK2;TK3];

% Plot results
% subplot(2,1,1)
% plot(time/60,Q)
% xlabel('Time (min)')
% ylabel('(0%-100%-0% heater)')
% subplot(2,1,2)
% plot(time/60.0,TK-273.15,'b-')
% ylabel('Temperature (degC)')
% xlabel('Time (min)')
% legend('Heat flow in time')


function dTdt = heat(time,x,Q,T0,U,m,Cp,A,beta,eps,sigma)
    
    %Parameters
    Ta = T0 + 273.15;   % K
    
    % Temperature State 
    T = x(1);

    % Nonlinear Energy Balance
    dTdt = (1/Cp)*(U*(A/m)*(Ta-T) ...
            + eps * sigma * (A/m) * (Ta^4 - T^4) ...
            + (beta/m)*Q);
end

end
