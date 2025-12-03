% Genera base de datos sintÃ©tica con puntos calientes siguiendo modelo de
% Stefan-Boltzmann y enfriamiento de Newton
clear;
close all;

ruta='/Users/David/Desktop/matlab_resisto/';

pathdata=[ruta '/volumes/matrices/']; %Datos
pathread=[ruta 'npy-matlab-master/npy-matlab/']; %lectura funcion readNPY
pathtemplate=[ruta 'volumes/masks/template.npy'];
pathout='/Users/David/Sourcecode/resisto-synthetic/data/original/';

generate_alarms = true;
generate_images = true;
images_path = '/Users/David/Sourcecode/resisto-synthetic/data/original/images-mat/';
prefix = 'SYN_CA1';


directorio=pwd;
cd(pathread)
template=readNPY(pathtemplate)';
cd(directorio)
tam=[384 288];
tam=[256 192];
% 

D=rdir([pathdata 'caminos/CT000001_2023-06-23/*.npy']); % Modelo entre semana
D_=rdir([pathdata 'caminos/CT000001_2023-06-24/*.npy']); % Model fin de semana

D=rdir([pathdata 'CT49025/CT49025_2023-10-18/*.npy']); % Modelo entre semana
D_=rdir([pathdata 'CT49025/CT49025_2023-10-29/*.npy']); % Model fin de semana
% D_=rdir([pathdata 'CT49025/CT49025_2023-10-18/*.npy']); % Model fin de semana


fecha=datetime(D(1).name(end-22:end-4),'InputFormat','yyyy-MM-dd_HH-mm-ss');

%% Código de comprobación para ver si está toda la secuencia de archivos
%
% for i = 1 : numel(D_)
%     minutos(i) = minute(datetime(D_(i).name(end-22:end-4),'InputFormat','yyyy-MM-dd_HH-mm-ss'));
% end
% 
% k = -1;
% idx = 0;
% 
% while true
%     
%     k = k + 1;
%     idx = idx + 1;
%     
%     minutos(idx) = minutos(idx) - k;
%     
%     if k == 59
%         k = -1;
%     end
%     
%     if idx == length(minutos)
%         break;
%     end   
% end

%% Load regions:

load('template-mask.mat')
template=image'-1;
I = template;
num_reg = max(max(template))+1;
%save('template.mat','template','mat_regions','fecha');
save('template.mat','template','fecha');

%% Extracting series.. one day...
 
for i=1:numel(D)
    cd(pathread);
    disp(['Extracting figure ' D(i).name ' de ' num2str(numel(D)-1)]) 
    It=readNPY(D(i).name);
    It_=readNPY(D_(i).name);
    cd(directorio)
    It=reshape(It,tam)';
    It_=reshape(It_,tam)';
    for nr=1:num_reg
        mat_sig(i,nr,1)=mean2(It(I'==nr-1));
        mat_sig(i,nr,2)=mean2(It_(I'==nr-1));
    end
    timeday(i)=datetime(D(i).name(end-22:end-4),'InputFormat','yyyy-MM-dd_HH-mm-ss');
    %datetime(Y,M,D,H,MI,S)
end

save('temp_one.mat','mat_sig','timeday');


%%
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
    
    Tmed_1=double(squeeze(mat_sig(:,selec_cluster,1)));
    Tmed_2=double(squeeze(mat_sig(:,selec_cluster,2)));
 
    % Vamos a eliminar los glitches de forma manual porque el filtro de
    % mediana no siempre funciona para distintos modelos 
    Tmed_1 = medfilt1(Tmed_1,7);
    Tmed_2 = medfilt1(Tmed_2,7);
  
    Numint=1;
    
    N=numel(Xtrain);
    T_op_1=[];
    T_op_2=[];
    
    for i=1:Numint
        
        step = floor(N/Numint);
        start = (i-1)*step+1;
        final = i*step;
        
        if i == Numint
            final = length(Xtrain);
        end
        
        Xtrains=Xtrain(start:final);
        Ytrains_1=Tmed_1(start:final);
        Ytrains_2=Tmed_2(start:final);
        
        [curve_1,gof_1] = fit(Xtrains,Ytrains_1,"poly5");
        [curve_2,gof_2] = fit(Xtrains,Ytrains_2,"poly5");
        
        T_op_1=[T_op_1; feval(curve_1,Xtrains)];
        T_op_2=[T_op_2; feval(curve_2,Xtrains)];
        
        % Temperaturas del sistema en un dia àra cada minuto T = Top + error(T amb)
        subplot(2,2,1)
        hold on;
        plot(timeday(round(start:final)),feval(curve_1,Xtrains));
        plot(timeday(round(start:final)),Ytrains_1)
        grid on
        subplot(2,2,3)
        hold on;
        plot(timeday(round(start:final)),feval(curve_2,Xtrains),timeday(round(start:final)),Ytrains_2);
        grid on
    end
    
   
    %Computing normalized distribution of errors
    
    error_1= Tmed_1-T_op_1; % Error de la aproximacion
    error_2= Tmed_2-T_op_2;
    
    pd_1 = fitdist(error_1,'Normal'); % distrbucion gaussiana N((Tmax-Tmin)mu,(Tmax-Tmin)sigma) hipotesis para incluir la dependencia con la T ambiente
    % de la parte dependiente de las condiciones de contorno T amb. Mayor
    % incremento de temperaturas mayor distribucion del error en el modelo.
    %Seleccionando T max y T min del dia
    pd_2 = fitdist(error_2,'Normal');
    
    
    fechastr=datestr(fecha,'dd-mmmm');
    fecha=datetime([fechastr '-2022']);
    [Tmax,Tmin]=find_temp_table(fecha,temperaturasG);
    
    % Recuperamos la distribucion normal original
    dispe=Tmax-Tmin;
    sigma_c_1=pd_1.sigma/dispe;     sigma_c_2=pd_2.sigma/dispe;
    mu_c_1=pd_1.mean/dispe;         mu_c_2=pd_2.mean/dispe;
    
    %histogram(error);
    %hold on;
    %% Extrapolating the whole year
    
    % Generation
    Tamb=(Tmax+Tmin)/2;
    
    for d=1:numel(dates)-1 % 365 dias del año
        
        %% Vamos si es fin de semana para elegir un modelo u otro:
        if ~isweekend(dates(d))
            mu_c = mu_c_1;
            sigma_c = sigma_c_1;
            T_op = T_op_1;
        else
            mu_c = mu_c_2;
            sigma_c = sigma_c_1;
            T_op = T_op_2;
        end
        
        % Corregimos la varianza del ruido para generar continuidad en los
        % días.
        correccion = 3;
        sigma_c = sigma_c * correccion;
        % Generamos valores de series en el dia
        Tmaxd=str2double(TsMax(d));
        Tmind=str2double(TsMin(d));
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
        
        gamma=alpha*emiss*sigma*((Tamb+273.15)^4-(Tambd+273.15)^4);
        %c = gamma; % You can adjust the value of c as needed
        % Generate some sample input values in the range [0, 1]
        %x = linspace(0, 1, numel(T_op));

        % Use the soft_mapping function to map values to the range [-c, 0]
        %correc = soft_mapping(x, c)';
        
        % Calculate the erf values for the given range   
        %if d==1
        T_op_new=((T_op+273.15)+gamma)-273.15;
        %else
        %    T_op_new=T_op+(T_op(end)-T_op(1));
        %    T_op_new=((T_op_new+273.15)+gamma+correc)-273.15;
        %end
        %T_op=T_op_new;
        %Tamb=Tambd;
        
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
        if generate_alarms
            palarm=0.05; % probability of alarm in each segment 35 dias al año con alarma
            if rand<palarm
                % NOTA: Modifico esto para que la alarma no salga al final
                % y se quede a medias como ocurre muchas veces
                % indt = randi([1 numel(Tmed)]);
                indt = randi([1 numel(Tmed)-40]);
                [~,TK]=heatfunction(Tmed(indt),U,m,Cp,A,beta,emiss,sigma); %simula sobre calientamiento
                Decimate=60*2; % Sampling factor 1 min * 2 min
                TKres=TK(1:Decimate:end);
                %Tmed=TKres; % aqui se asume que la cámara capta el calentamiento 1 a 1
                Tmednew=(Tmed(indt)+273.15)+alpha*sigma*emiss*(TKres.^4-(Tambd+273.15)^4); % error nulo en comparacion con la anomalia
                %(Tmed(indt)+273.15) hace el papel de Top
                
                % NOTA: AQUÍ HE CAMBIADO TmednewC=Tmednew-273.15 PORQUE ME
                % SUBÍA LA ANOMALIA VARIOS GRADOS Y SALÍA MAL.
                TmednewC=TKres-273.15; %Celsius

                % La anomalia empezará cuando empiece a aumentar el
                % calentamiento y no antes...
                start_anomaly = find(diff(TmednewC));
                indt = indt + start_anomaly(1);
                
                % NOTA: Aquí modifico el indice final de la anomalía porque
                % ahora nunca sobrepasará el final de la señal diaria.
                % idxf=min([indt+numel(TmednewC)-1 numel(Tmed)]);

                idxf= numel(TmednewC);

                Tmed(indt:indt+idxf-start_anomaly(1))=TmednewC(start_anomaly(1):idxf);
                            subplot(1,2,2);
                            plot(timeday,Tmed);
                            grid on
                            title(['cluster ' num2str(numcl)])
                alarms(numcl,d,indt:indt+idxf-start_anomaly(1))=1; %create alarm variable
            end
        end
 %% Save data       
        datap(numcl,d,:)=Tmed;
        % Salvamos la serie 
        
    end
    
    
end

if generate_alarms
    save([pathout 'alarms-mat/alarms.mat'],'alarms');
    saving_folder = 'temperatures-alarms-mat/';
else
    saving_folder = 'temperatures-mat/';
end

%% Salvamos Series
for d=1:numel(dates)-1
    
    Tmed=squeeze(datap(:,d,:));
    
    year = num2str(dates.Year(d));
    month = num2str(dates.Month(d),'%02.f');
    day = num2str(dates.Day(d),'%02.f');
    
    disp(['Saving ' prefix '_'  year '-' month '-' day])
    save([pathout saving_folder prefix '_' year '-' month '-' day '.mat'],'Tmed')
end


%% Guardamos la secuencia de imágenes
%  Todos los píxeles de una región se igualarán a la temperatura media de
%  esa zona calculada por el código anterior.

% Cargamos la máscara de regiones:
load('template.mat');

% Generamos las imágenes para cada día:
if generate_images
    for d=1:numel(dates)-1
        disp(['Saving images for date ' datestr(dates(d))])
        Idates=zeros(size(image));
        
        % Para cada imagen:
        for t=1:numel(T_op)
            
            datet=D(t).name(end-11:end-4);
            
            % Para cada region:
            for k=1:size(mat_sig,2)
                Idates(image == k) = datap(k,d,t);
            end
            
            year = num2str(dates.Year(d));
            month = num2str(dates.Month(d),'%02.f');
            day = num2str(dates.Day(d),'%02.f');
            
            folder_name = [filesep prefix '_' year '-' month '-' day filesep];
            image_name = [prefix '_' year '-' month '-' day '_' datet '.mat'];
            
            
            % Formato CT000001_CA1_2023-06-17_23-42-08
            try
                save([images_path folder_name image_name],'Idates')
            catch ME
                mkdir([images_path folder_name]);
                save([images_path folder_name image_name],'Idates')
            end
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

function y = soft_mapping(x, c)
    % Check if x is in the interval [0, 1]
    if any(x < 0) || any(x > 1)
        error('Input values must be in the range [0, 1]');
    end
    
    % Ensure c is positive
    if c > 0
        y = -c * (1 - x.^2);
        
    else
        y = c - c * x.^2;
    end
    
   
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
    dTdt = (1/Cp)*(U*(A/m)*(Ta-T) + eps * sigma * (A/m) * (Ta^4 - T^4) + (beta/m)*Q);
end



end
