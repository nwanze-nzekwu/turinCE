

function RytovVar = rytov_vs_latency(Pow)
    visualDebug = false;
    Pow_w = Pow; % pow is watts, not db!

    %% Params

    [~, nn] = size(Pow_w);
    nSamples = length(Pow_w);
    Fs = 10e3;
    T = 1/Fs * nSamples;
    t = linspace(0, T, nSamples);

    %%
    % Convert Received Optical Power to Irradiance
    d = 7e-3; % F810APC
    I_meas = power2irradiance(Pow_w, d);

    for l = 1:nn
        I_meas(:, l) = I_meas(:, l) ./ movmean((I_meas(:, l)), 1000);
    end

    I_fastFading = I_meas;

    %% Get Histogram

    clear RytovVar
    nBins = 1.5e5;
    yTh = 1e-6;
    T_hist_min = 1; % tempo a considerar por cada block

    nSamples_block = length(I_fastFading); % Process entire dataset as a single block
    nBlocks = nSamples / nSamples_block;

    for l = 1:nn
        for n = 1:nBlocks
            I_fit = I_fastFading((n-1)*nSamples_block+1:n*nSamples_block, l);

            %% Small test neves
            % I_fit(I_fit>10*mean(I_fit))=nan;
            %%
            [N, edges] = histcounts(I_fit, nBins);
            xx = edges(1:end-1);
            yy = N / sum(N);

            xx1 = xx(yy > yTh);
            yy1 = yy(yy > yTh);

            % Normalize to Unitary Integral Area:
            y{n} = yy1 / trapz(xx1, yy1);
            x{n} = xx1;
        end

        % Estimate Rytov Variance By Fitting Against Model
        MSE = @(x, y) nanmean(abs(x-y).^2);

        RV_init = 0.08;

        for n = 1:nBlocks
            % Initial fit with Gamma-Gamma
            costFunGammaGamma = @(RV, x, y) MSE(pdf_GammaGamma_wrapped(RV, x), y);
            options = optimset('MaxFunEvals', 1e5, 'TolX', 1e-10, 'TolFun', 1e-10, ...
                'Display', 'none', 'PlotFcns', []);
            [RytovVarGammaGamma, MSE_valGammaGamma] = fminsearch(@(RV) costFunGammaGamma(RV, x{n}, y{n}), ...
                RV_init, options);

            % Check Rytov value and switch to Log-Normal if below 0.03
            if RytovVarGammaGamma < 0.03
                costFunLogNormal = @(RV, x, y) MSE(pdf_LogNormal(RV, x), y);
                [RytovVar(n, l), MSE_val(n, l)] = fminsearch(@(RV) costFunLogNormal(RV, x{n}, y{n}), ...
                    RV_init, options);
                disp('Switched to log-normal...')
            else
                RytovVar(n, l) = RytovVarGammaGamma;
                MSE_val(n,l) = MSE_valGammaGamma;
            end
        end

        if visualDebug
            try
                close(hFig)
            end
            hFig = figure();
            plot(x{1}, y{1}, 'DisplayName', 'actual pdf');
            hold on
            if RytovVar(1, l) < 0.03 % or whatever threshold value
                plot(x{1}, pdf_LogNormal(RytovVar(1, l), x{1}), 'DisplayName', 'lognormal pdf');
            else
                plot(x{1}, pdf_GammaGamma_wrapped(RytovVar(1, l), x{1}), 'DisplayName', 'Gamma-Gamma pdf');
            end
            legend();
        end
        clear x y
    end
end























% 
% function RytovVar   = rytov_vs_latency(Pow)
% 
%     visualDebug = false;
% 
%     Pow_w = Pow; % pow is watts, not db!
%     
%     %% Params
%     
%     [~, nn] = size(Pow_w);
%     nSamples= length(Pow_w);
%     Fs = 10e3;
%     T = 1/Fs * nSamples;
%     t =linspace(0,T,nSamples);
%     
%     %%
%     %Convert Received Optical Power to Irradiance
%     d = 7e-3; % F810APC
%     I_meas = power2irradiance(Pow_w,d);
%     
%     for l=1:nn
%         I_meas(:,l) = I_meas(:,l) ./ movmean((I_meas(:,l)),1000);    
%     end
%     
%     I_fastFading =I_meas;
%     
%     %% Get Histogram
%     
%     clear RytovVar
%     nBins = 1.5e5;
%     yTh = 1e-6;
%     T_hist_min = 1; % tempo a considerar por cada block
%     
% %     nSamples_block = 60 * T_hist_min / (1/Fs);
%     nSamples_block = length(I_fastFading); % Process entire dataset as a single block
%     nBlocks = nSamples / nSamples_block;
% 
%         
%     for l=1:nn
%         for n = 1:nBlocks
% 
%             I_fit = I_fastFading((n-1)*nSamples_block+1:n*nSamples_block,l);
%             
% 
%             %% Small test neves
% %             I_fit(I_fit>10*mean(I_fit))=nan;
%             %%
% 
%             [N,edges] = histcounts(I_fit,nBins);
%             xx = edges(1:end-1);
%             yy = N / sum(N);       
%             
%             xx1 = xx(yy > yTh);
%             yy1 = yy(yy > yTh);
%             
%             % Normalize to Unitary Integral Area:
%             y{n} = yy1 / trapz(xx1,yy1);
%             x{n} = xx1;        
%         end
%         
%         % Estimate Rytov Variance By Fitting Against Log-Normal Model
%         MSE = @(x,y) nanmean(abs(x-y).^2);  
%         
% %         costFun = @(RV,x,y) MSE(pdf_LogNormal(RV,x),y); % Define Cost Function To Be Minimized:
%         costFun = @(RV,x,y) MSE(pdf_GammaGamma_wrapped(RV,x),y);
% 
%         
% 
%         options = optimset('MaxFunEvals',1e5,'TolX',1e-10,'TolFun',1e-10,...
%             'Display','none','PlotFcns',[]); % Optimize @optimplotfval
%         
%         RV_init = 0.08;
%     
%         for n = 1:nBlocks
%             [RytovVar(n,l),MSE_val(n,l)] = fminsearch(@(RV) costFun(RV,x{n},y{n}),...
%                RV_init,options);
%         end
%         
%         if visualDebug
%             try 
%                 close(hFig) 
%             end
%             hFig = figure();
%             plot(x{1},y{1},displayname='actual pdf')
%             hold on
%             plot(x{1},pdf_LogNormal(RytovVar,x{1}),displayname='lognormal pdf') 
%             legend();
%         end
% 
%         clear x y
% 
%     end
% 
% end
