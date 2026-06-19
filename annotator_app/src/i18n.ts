/* Minimal i18n: English / Mandarin (Simplified). `tr(lang, key)` returns the string; components read
   the current `lang` from the store. Dynamic status messages generated in the store stay in English. */

export type Lang = "en" | "zh";

const T = {
  // Login gate
  "login.title":      { en: "Cornea Ground-Truth Annotator", zh: "角膜金标准标注工具" },
  "login.subtitle":   { en: "Manual scar segmentation", zh: "手动瘢痕分割" },
  "login.desc":       { en: "Choose who is annotating. Your username is recorded with every saved label — enabling inter-observer (different people) and intra-observer (same person, different sessions) analysis.",
                        zh: "请选择标注者。您的用户名会记录在每次保存的标注中——用于观察者间（不同人）与观察者内（同一人、不同时段）一致性分析。" },
  "login.existing":   { en: "Existing user", zh: "已有用户" },
  "login.select":     { en: "Select a user…", zh: "选择用户…" },
  "login.enter":      { en: "Enter", zh: "进入" },
  "login.orAdd":      { en: "or add a new user", zh: "或添加新用户" },
  "login.addBegin":   { en: "add a user to begin", zh: "添加用户以开始" },
  "login.newUser":    { en: "New user", zh: "新用户" },
  "login.username":   { en: "username", zh: "用户名" },
  "login.addEnter":   { en: "Add & enter", zh: "添加并进入" },

  // Header / save bar
  "app.title":        { en: "Ground-Truth Annotator", zh: "金标准标注工具" },
  "app.subtitle":     { en: "Cornea scar segmentation", zh: "角膜瘢痕分割" },
  "save.userTip":     { en: "The active annotator — recorded with every saved label for inter-/intra-observer analysis",
                        zh: "当前标注者——记录于每次保存，用于观察者间/内分析" },
  "save.switch":      { en: "switch", zh: "切换" },
  "save.switchTip":   { en: "Switch to a different annotator", zh: "切换到其他标注者" },
  "save.setOutput":   { en: "Set output folder…", zh: "设置输出文件夹…" },
  "save.outputTip":   { en: "Choose where annotations are saved", zh: "选择标注的保存位置" },
  "save.save":        { en: "Save ground truth", zh: "保存金标准" },
  "save.saveTip":     { en: "Write the painted labelmap (0/1/2) + a manifest row tagged with your username and this session",
                        zh: "写出绘制的标签图（0/1/2），并附带含用户名与本次会话的清单记录" },
  "updates.label":    { en: "Updates", zh: "更新" },
  "updates.checking": { en: "Checking…", zh: "检查中…" },
  "updates.tip":      { en: "Check for a newer version and install it in-app", zh: "检查并在应用内安装新版本" },
  "updates.latest":   { en: "You're on the latest version", zh: "已是最新版本" },
  "about.label":      { en: "About", zh: "关于" },
  "about.madeBy":     { en: "Made by Zhuojian Look", zh: "由 Zhuojian Look 制作" },
  "about.desc":       { en: "A cross-platform tool for human ground-truth scar segmentation on preprocessed OCT volumes.",
                        zh: "用于在预处理 OCT 体数据上进行人工金标准瘢痕分割的跨平台工具。" },
  "about.version":    { en: "Version", zh: "版本" },
  "about.close":      { en: "Close", zh: "关闭" },
  "lang.name":        { en: "Language", zh: "语言" },

  // Paint toolbar
  "tb.paint":     { en: "Paint", zh: "绘制" },
  "tb.navigate":  { en: "Navigate", zh: "导航" },
  "tb.pen":       { en: "Pen", zh: "画笔" },
  "pen.cornea":   { en: "Cornea", zh: "角膜" },
  "pen.scar":     { en: "Scar", zh: "瘢痕" },
  "pen.erase":    { en: "Erase", zh: "擦除" },
  "tb.size":      { en: "Size", zh: "大小" },
  "tb.sizeTip":   { en: "Brush size (voxels)", zh: "画笔大小（体素）" },
  "tb.fill":      { en: "Fill region", zh: "填充区域" },
  "tb.fillTip":   { en: "Filled pen: draw a closed outline → fill the enclosed region (one stroke per patch).",
                    zh: "填充笔：画出闭合轮廓 → 填充其内部区域（每块一次描边）。" },
  "tb.smartFill": { en: "Smart fill", zh: "智能填充" },
  "tb.smartTip":  { en: "Smart fill (GrowCut): scribble a little Cornea AND Scar on a few slices, then propagate through the whole 3-D volume by intensity similarity — so you don't paint every slice.",
                    zh: "智能填充（GrowCut）：在若干切片上分别涂少量角膜与瘢痕，再按强度相似度扩展到整个三维体——无需逐片绘制。" },
  "tb.undo":      { en: "Undo", zh: "撤销" },
  "tb.clear":     { en: "Clear", zh: "清除" },
  "tb.opacity":   { en: "Opacity", zh: "不透明度" },
  "tb.opacityTip":{ en: "Label overlay opacity", zh: "标签叠加不透明度" },

  // Volume browser
  "vol.volumes":  { en: "Volumes", zh: "体数据" },
  "vol.done":     { en: "done", zh: "完成" },
  "vol.pick":     { en: "Pick folder of NIfTI…", zh: "选择 NIfTI 文件夹…" },
  "vol.change":   { en: "Change folder…", zh: "更换文件夹…" },
  "vol.noFolder": { en: "No folder selected", zh: "未选择文件夹" },
  "vol.noFolderHint": { en: "Pick a folder of preprocessed .nii / .nii.gz volumes to begin.",
                        zh: "请选择包含预处理 .nii / .nii.gz 体数据的文件夹以开始。" },
  "vol.noFiles":  { en: "No .nii / .nii.gz files in this folder.", zh: "此文件夹中没有 .nii / .nii.gz 文件。" },

  // Canvas
  "view.multi":    { en: "Multi", zh: "多视图" },
  "view.axial":    { en: "Axial", zh: "轴位" },
  "view.coronal":  { en: "Coronal", zh: "冠状" },
  "view.sagittal": { en: "Sagittal", zh: "矢状" },
  "view.render":   { en: "3D", zh: "三维" },
  "canvas.noVolume":     { en: "No volume loaded", zh: "未加载体数据" },
  "canvas.noVolumeHint": { en: "Pick a folder and choose a volume on the left to start annotating.",
                           zh: "在左侧选择文件夹与体数据以开始标注。" },
  "canvas.loading": { en: "Loading volume…", zh: "正在加载体数据…" },
  "canvas.noVol":   { en: "no volume", zh: "无体数据" },
} as const;

export type TKey = keyof typeof T;

export function tr(lang: Lang, key: TKey): string {
  const entry = T[key];
  return (entry && (entry[lang] ?? entry.en)) || key;
}
