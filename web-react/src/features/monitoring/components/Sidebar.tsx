import { useState, type ComponentProps, type ReactNode } from 'react';
import Icon from '../../../components/Icon';
import type { GazePrediction } from '../../gaze/hooks/useGazePrediction';

interface Props {
  cameraOn: boolean;
  cameraLoading: boolean;
  modelLoading: boolean;
  analyzing: boolean;
  resetting: boolean;
  serviceOnline: boolean;
  serviceText: string;
  gaze: GazePrediction;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onCamera: () => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onCapture: () => void;
}

function ControlGroup({ id, label, icon, children, className = '' }: {
  id: string; label: string; icon: ComponentProps<typeof Icon>['name']; children: ReactNode; className?: string;
}) {
  const [expanded, setExpanded] = useState(true);
  return <section className={`control-group ${className}`}>
    <button type="button" className="group-toggle" aria-expanded={expanded} aria-controls={id}
      aria-label={label} title={`${expanded ? '收起' : '展开'}${label}`} onClick={() => setExpanded(value => !value)}>
      <Icon name={icon} /><span className="group-label">{label}</span>
      <span className={`group-chevron${expanded ? ' expanded' : ''}`}><Icon name="chevron" /></span>
    </button>
    <div id={id} className="group-content" hidden={!expanded}>{children}</div>
  </section>;
}

function SideButton({ icon, children, className = '', ...props }: ComponentProps<'button'> & {
  icon: ComponentProps<typeof Icon>['name'];
}) {
  return <button type="button" className={`side-button ${className}`} {...props}>
    <Icon name={icon} /><span className="button-label">{children}</span>
  </button>;
}

export default function Sidebar(props: Props) {
  const { gaze } = props;
  const cameraLabel = props.cameraLoading ? '正在开启摄像头' : props.cameraOn ? '关闭摄像头' : '开启摄像头';
  const analysisLabel = props.modelLoading ? '正在加载模型' : props.analyzing ? '识别运行中' : '开始识别';
  const gazeOn = gaze.status === 'running' || gaze.status === 'loading';
  const gazeLabel = gaze.status === 'loading' ? '加载中' : gazeOn ? '已开启' : '已关闭';

  return <aside className="sidebar" aria-label="功能控制栏">
    <div className="brand">
      <div className="brand-mark" aria-hidden="true">FL</div>
      <div className="brand-text"><strong>FocusLens</strong><span>视觉状态实验台</span></div>
      <button className="sidebar-toggle" type="button" onClick={props.onToggleCollapsed}
        aria-label={props.collapsed ? '展开侧边栏' : '收起侧边栏'} title={props.collapsed ? '展开侧边栏' : '收起侧边栏'}
        aria-expanded={!props.collapsed}><Icon name="chevron" /></button>
    </div>
    <div className={`sidebar-status${props.serviceOnline ? ' online' : ''}`} title={props.serviceText}>
      <span className="status-dot" />
      <div><strong>{props.serviceText}</strong><p>情绪与眨眼识别服务</p></div>
    </div>
    <nav className="controls" aria-label="识别控制">
      <div className="controls-main">
      <ControlGroup id="capture-controls" label="采集控制" icon="camera">
        <SideButton icon="camera" className={props.cameraOn ? '' : 'primary'} onClick={props.onCamera}
          disabled={props.cameraLoading} aria-label={cameraLabel} title={cameraLabel}>{cameraLabel}</SideButton>
        <SideButton icon="play" className={props.analyzing ? 'active' : ''} onClick={props.onStart}
          disabled={!props.cameraOn || props.analyzing || props.modelLoading || props.resetting}
          aria-label={analysisLabel} title={analysisLabel}>{analysisLabel}</SideButton>
        <SideButton icon="pause" onClick={props.onPause} disabled={!props.analyzing}
          aria-label="暂停分析" title="暂停分析">暂停分析</SideButton>
      </ControlGroup>
      <ControlGroup id="gaze-controls" label="视线预测" icon="eye">
        <button type="button" className="gaze-switch" role="switch" aria-label="视线预测"
          aria-checked={gazeOn} disabled={!props.cameraOn} title={`视线预测 · ${gazeLabel}`}
          onClick={() => gazeOn ? gaze.pause() : void gaze.start()}>
          <span className="button-label">视线预测</span><span className="switch-track" aria-hidden="true" />
          <span className="switch-status">{gazeLabel}</span>
        </button>
        <SideButton icon="target" onClick={gaze.beginCalibration} disabled={gaze.status !== 'running'}
          aria-label={gaze.calibrated ? '重新校准视线' : '开始视线校准'} title={gaze.calibrated ? '重新校准视线' : '开始视线校准'}>
          {gaze.calibrated ? '重新校准视线' : '开始视线校准'}
        </SideButton>
        <SideButton icon="fullscreen" onClick={() => void gaze.toggleFullscreen()} aria-label="切换全屏" title="切换全屏">切换全屏</SideButton>
        <label className="point-toggle" title="显示落点"><input type="checkbox" checked={gaze.showPoint}
          aria-label="显示落点" onChange={event => gaze.setShowPoint(event.target.checked)} /><span className="button-label">显示落点</span></label>
      </ControlGroup>
      <div className="privacy-note" title="本地处理：画面仅在本机分析，不保存视频。">
        <Icon name="shield" /><div><strong>本地处理</strong><p>画面仅在本机分析，不保存视频。</p></div>
      </div>
      </div>
      <ControlGroup id="tool-controls" label="工具" icon="tools" className="tools-group">
        <SideButton icon="image" onClick={props.onCapture} disabled={!props.cameraOn} aria-label="保存截图" title="保存截图">保存截图</SideButton>
        <SideButton icon="reset" onClick={props.onReset} disabled={props.resetting || props.modelLoading} aria-label="重置统计" title="重置统计">
          {props.resetting ? '正在重置' : '重置统计'}
        </SideButton>
      </ControlGroup>
    </nav>
  </aside>;
}
