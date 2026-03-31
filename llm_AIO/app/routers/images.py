from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.multimodal import ImageGenRequest
from app.adapters.aliyun_image import AliyunImageAdapter
from app.adapters.doubao_image import DoubaoImageAdapter
from app.database import get_convert_db
from app.services.file_processing import convert_base64_to_url
# from app.adapters.local_image import LocalImageAdapter

router = APIRouter(prefix="/api/playground/images", tags=["images"])

def _is_base64_data(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")

@router.post("/generations")
async def generate_image(request: ImageGenRequest, db: Session = Depends(get_convert_db)):
    provider_clean = request.provider.strip().lower() if request.provider else ""

    # 非本地模型：将上传的 base64 转为 URL 再传给下游
    if provider_clean != "local" and request.image:
        if isinstance(request.image, list):
            request.image = [
                await convert_base64_to_url(item, db, default_ext="png") if _is_base64_data(item) else item
                for item in request.image
            ]
        elif _is_base64_data(request.image):
            request.image = await convert_base64_to_url(request.image, db, default_ext="png")

    if provider_clean == "aliyun":
        adapter = AliyunImageAdapter()
        return await adapter.generate(request)
    elif provider_clean == "doubao":
        adapter = DoubaoImageAdapter()
        return await adapter.generate(request)
    elif provider_clean == "local":
        # adapter = LocalImageAdapter()
        # return await adapter.generate(request)
        return {"error": "Local image generation not yet implemented"}
    else:
        raise HTTPException(status_code=400, detail="Unknown provider")

# --- Educational Prompt Templates ---
# --- Educational Prompt Templates (University Disciplines) ---
PROMPT_TEMPLATES = [
    # -------------------------------------------------------------------------
    # 1. 专业课件可视化类 (Specialized Courseware Visualization)
    # -------------------------------------------------------------------------
    {
        "id": "stem_structure_viz",
        "category": "专业课件可视化 (Specialized Courseware)",
        "title": "神经网络与算法结构",
        "description": "生成神经网络或算法（如CNN, Transformer）的结构示意图。",
        "prompt": "大学人工智能专业课件插图，Transformer模型结构示意图，风格为矢量扁平化，配色蓝白灰冷色调，画面包含输入嵌入层、多头注意力层、前馈神经网络层、残差连接与层归一化模块，标注中英文专业术语，箭头标注序列数据流向，背景纯白，分辨率2K，适合本科《深度学习导论》课堂使用。",
        "negative_prompt": "模糊，杂乱，错误的连接，手绘风格，低分辨率，水印，多余装饰",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    },
    {
        "id": "stem_experiment_logic",
        "category": "专业课件可视化 (Specialized Courseware)",
        "title": "实验装置/模型逻辑图",
        "description": "生成理工科实验装置或经管类模型逻辑图。",
        "prompt": "大学金融专业课件插图，资本资产定价模型（CAPM）逻辑图，风格为简约商务风，配色蓝金，画面包含无风险收益率、市场风险溢价、贝塔系数、预期收益率四大模块，箭头标注变量运算关系，背景浅灰渐变，适合专硕《投资学》课堂使用。",
        "negative_prompt": "卡通，甚至恐怖，复杂背景，不可读文字，乱涂乱画",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    },

    # -------------------------------------------------------------------------
    # 2. 教学素材优化类 (Teaching Material Optimization)
    # -------------------------------------------------------------------------
    {
        "id": "edu_face_swap_prompt",
        "category": "教学素材优化 (Teaching Material)",
        "title": "教学案例人物换脸",
        "description": "为教学案例生成特定角色的人物图像（提示词模板）。",
        "prompt": "教育素材人物换脸生成，仅用于大学《金融科技伦理》教学演示，无商用意图。源图特征：站立在报告厅演讲的男性，侧身抬手姿态；目标人物特征：戴黑框眼镜的中年教授，短发，深蓝色西装；保留源图姿态与舞台光影，换脸后面部表情自然，与身体比例协调无违和感，背景保留报告厅场景，分辨率2K，适合课程案例课件使用。",
        "negative_prompt": "扭曲的面部，不协调的身体比例，模糊，不像人，恐怖谷",
        "config": {
            "style": "photography",
            "size": "1024x1024",
            "prompt_extend": True
        }
    },
    {
        "id": "edu_bg_replace_prompt",
        "category": "教学素材优化 (Teaching Material)",
        "title": "配图背景更换",
        "description": "为课件或论文配图生成背景更换的提示词。",
        "prompt": "大学计算机专业论文配图背景更换，源图主体：YOLOv8目标检测算法热力图，保留热力图颜色梯度与细节；目标背景：符合《计算机学报》规范的纯白背景，无网格无杂物，主体边缘清晰无虚化，分辨率300dpi，适合学术论文发表使用。",
        "negative_prompt": "杂乱背景，水印，模糊主体，有色差",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    },

    # -------------------------------------------------------------------------
    # 3. 科研成果呈现类 (Research Results Presentation)
    # -------------------------------------------------------------------------
    {
        "id": "res_chart_optimization",
        "category": "科研成果呈现 (Research Results)",
        "title": "学术图表优化",
        "description": "优化科研图表的美观度和规范性。",
        "prompt": "大学金融科技专业学术图表优化，图表类型为柱状图，主题为\"区块链跨境支付与传统支付效率对比\"，配色为《金融研究》期刊标准蓝灰配色，横轴标注支付类型，纵轴标注交易耗时（单位：秒），图例明确，背景白底，分辨率300dpi，适合学术论文发表。",
        "negative_prompt": "手绘，非正规图表，乱码，低分辨率，花哨",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    },
    {
        "id": "res_lab_scene",
        "category": "科研成果呈现 (Research Results)",
        "title": "科研场景还原",
        "description": "还原实验室或调研现场场景。",
        "prompt": "大学生物工程专业科研场景还原图，主题为基因编辑CRISPR-Cas9实验，风格为写实风，画面包含穿白大褂的科研人员、PCR仪、基因测序仪、培养皿，人物动作是观察培养皿样本，设备细节符合分子生物实验标准，背景为实验室操作台，色调理性灰白，分辨率300dpi，适合科研项目汇报。",
        "negative_prompt": "科幻夸张，非专业设备，错误的操作，脏乱",
        "config": {
            "style": "photography",
            "size": "1024x1024",
            "prompt_extend": True
        }
    },

    # -------------------------------------------------------------------------
    # 4. 新增高频场景类 (High Frequency Scenarios)
    # -------------------------------------------------------------------------
    {
        "id": "scene_ppt_material",
        "category": "高频场景 (High Frequency)",
        "title": "答辩PPT素材",
        "description": "生成适用于学术答辩PPT的背景或配图。",
        "prompt": "大学金融科技专业毕业答辩PPT封面底图，主题为\"基于YOLO算法的金融票据伪造检测研究\"，风格为科技感商务风，配色蓝白灰，画面包含芯片、票据图案、数据流线条，上方留白用于填写论文标题与答辩人信息，尺寸16:9，适合硕士毕业答辩使用。",
        "negative_prompt": "复杂文字，干扰阅读的图案，低俗",
        "config": {
            "size": "1920x1080",
            "prompt_extend": True
        }
    },
    {
        "id": "scene_sim_experiment",
        "category": "高频场景 (High Frequency)",
        "title": "虚拟仿真实验",
        "description": "生成虚拟仿真实验的示意图。",
        "prompt": "大学计算机网络专业虚拟仿真实验示意图，实验主题为\"TCP/IP协议路由转发仿真\"，风格为3D建模风，配色科技蓝，画面包含路由器、交换机、终端主机、数据传输路径，标注路由表配置步骤，背景为仿真机房环境，分辨率2K，适合虚拟仿真课程教学。",
        "negative_prompt": "平面图，低质感，错误连接",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    },
    {
        "id": "scene_competition_poster",
        "category": "高频场景 (High Frequency)",
        "title": "竞赛宣传素材",
        "description": "生成学科竞赛的宣传海报素材。",
        "prompt": "大学金融科技挑战赛宣传海报底图，风格为活力科技风，配色蓝橙，画面包含金融K线图、芯片、代码流元素，中间留白用于填写竞赛名称与报名信息，尺寸16:9横版，适合金融学院公众号宣传。",
        "negative_prompt": "文字过多，布局杂乱，水印",
        "config": {
            "size": "1920x1080",
            "prompt_extend": True
        }
    },
    {
        "id": "scene_general_edu",
        "category": "高频场景 (High Frequency)",
        "title": "通识教育插图",
        "description": "生成通识教育课程的互动插图。",
        "prompt": "大学通识教育互动插图，主题为\"经济学博弈论之囚徒困境\"，风格为漫画分镜，配色明快黄蓝，画面分两格：左格是两个囚徒的选择场景，右格是收益矩阵通俗解释，文字标注\"坦白 vs 沉默的利益权衡\"，分辨率2K，适合《生活中的经济学》通识课互动使用。",
        "negative_prompt": "抽象难懂，枯燥，甚至恐怖，血腥",
        "config": {
            "size": "1024x1024",
            "prompt_extend": True
        }
    }
]

@router.get("/prompts/templates")
async def get_prompt_templates():
    """
    Get a list of educational prompt templates.
    """
    return {
        "count": len(PROMPT_TEMPLATES),
        "templates": PROMPT_TEMPLATES
    }
