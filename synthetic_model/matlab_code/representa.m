% Representa
clear 
close all
ruta='/Users/juanma/Documentos/GESTION/Proyecto RESISTO Endesa/matlab_resisto/';

pathname=[ruta 'data/images_alarm/'];
pathbis=[ruta 'data/series_alarm/'];
load('temperaturasG.mat'); % carga temperaturasG Ambientes
load('temp_one.mat'); % carga mat_sig y time
load([ruta 'data/series_alarm/alarms.mat']);
dates=[temperaturasG.FECHA; temperaturasG.FECHA1;temperaturasG.FECHA2];
load('template.mat');
%fechastr=datestr(fecha,'dd-mmmm');
%fecha=datetime([fechastr '-2022']);



imagesc(template');
colorbar

selec_cluster=7;
[i,j]=find(squeeze(alarms(selec_cluster,:,:))>0);
fecha_alarm=dates(i(1));

timedaystr=datestr(timeday,'HH:MM:SS');
timedayd=datetime([repmat(datestr(fecha_alarm),numel(timeday),1) repmat(' ',numel(timeday),1) timedaystr],'InputFormat','dd-MMM-yyyy HH:mm:ss');

pathname=[pathname num2str(month(fecha_alarm)) '/' num2str(day(fecha_alarm)) '/'];
D=dir([pathname]);

%%
figure
symbols={'*','o','.','+','v','x','s','d','<','>'};
linestyle={'-','--',':','-.','-','--',':','-.','-','--',':','-.'};
%'LineStyle',linestyle{i},'Marker',symbols{i},
load([pathname D(3).name]);
f1=subplot(1,2,1);
imagesc(Idates');
axis off
c=colormap(f1,jet);
colorbar('Ticks',[0,10,15,20,30,40,50,60],...
         'TickLabels',{'0','10','15','20','30','40','50','60'})     
cont=1;
v = VideoWriter('evolucion.mp4','MPEG-4');
open(v)

f2=subplot(1,2,2)
load([pathbis 'CT000001_CA1_2022-' num2str(month(fecha_alarm)) '-' num2str(day(fecha_alarm)) '.mat']);
h=plot(timedayd,seriest','LineWidth',2);
set(h, {'color'}, num2cell(jet(9), 2));
%set(h, {'color'}, c);
%colormap(f2,c)

% for i=1:max(size(h))
%     set(h(i),'LineStyle',linestyle{i})
% end
h=legend(h,{'Floor','Transformer 2','Fence','Background','Bushings','Tank cover down','Tank cover up 1','Tank cover up 2','Tank cover up 3'});
h.AutoUpdate = 'off';

hold on
%plot(ones(numel((25:1:50)),1),(25:1:50)','k-.','LineWidth',5);
ytime=(25:(50-25)/(numel(timeday)-1):50)';
ytime=(25:1:50)';
h2=plot(repmat(timedayd(3),numel(ytime),1),ytime,'k-.','LineWidth',5);

for i=4:4:numel(D)
    subplot(1,2,1)
    hold on
    load([pathname D(i).name]);
    imagesc(Idates')
    frame = getframe(gcf);
    cont=cont+1;
    pause(0.1)
    writeVideo(v,frame);
    subplot(1,2,2)
    hold on
    %plot(i*ones(numel((25:1:50)),1),(25:1:50)','k-.','LineWidth',5)
    delete(h2)
    h2=plot(repmat(timedayd(i-2),numel(ytime),1),ytime,'k-.','LineWidth',5);

    hold off
end


close(v)